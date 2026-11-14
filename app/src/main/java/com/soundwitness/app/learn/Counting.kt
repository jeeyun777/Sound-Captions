package com.soundwitness.app.learn

import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.audio.SAMPLE_RATE
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Counting-based sound model (TCAM-style).
 *
 * A 1-second segment becomes a 28 (frequency) x 28 (time) image of band energies,
 * which is binarized. Every 3x3 neighbourhood of that binary image is a 9-bit
 * pattern. A model is just a table of how often each pattern appeared in each
 * frequency region. Learning = adding counts. Scoring = looking counts up.
 * No multiplication-heavy training and no backpropagation.
 */
object Encoder {
    const val SIZE = 28
    var REL_DB = 6f
    const val REGIONS = 7
    const val CODES = 512
    const val FEATURES = REGIONS * CODES
    val edges: IntArray = Dsp.bandEdges(SIZE)

    class Encoded(val image: Array<FloatArray>, val codes: IntArray, val bandMeanDb: FloatArray)

    /**
     * [ref] = this car's usual background level per band (dB). When given, cells are compared
     * with the car's own background first, so the pattern shows what stands out from it
     * rather than what is simply loud. Either way the cut is relative (top 30%), so a
     * louder or quieter microphone gives the same pattern.
     */
    fun encode(pcm: ShortArray, start: Int = 0, length: Int = SAMPLE_RATE, ref: FloatArray? = null): Encoded {
        val frames = Dsp.bandFrames(pcm, start, length, edges)
        val img = Dsp.poolTime(frames, SIZE)            // [band][time], low band first
        val bandMean = FloatArray(SIZE) { b -> img[b].average().toFloat() }
        val rel = if (ref == null) img else Array(SIZE) { b -> FloatArray(SIZE) { t -> img[b][t] - ref[b] } }
        // binarize: a cell is 1 when it is in the top 30% of this (background-relative) image
        val all = rel.flatMap { it.asIterable() }.sorted()
        val q = all[(all.size * 0.70).toInt().coerceIn(0, all.size - 1)]
        // against a background profile a cell must also stand REL_DB above that background
        val thr = if (ref == null) q else maxOf(q, REL_DB)
        val bits = Array(SIZE) { b -> BooleanArray(SIZE) { t -> rel[b][t] >= thr } }
        val codes = IntArray((SIZE - 2) * (SIZE - 2))
        var n = 0
        for (b in 1 until SIZE - 1) {
            val region = (b * REGIONS) / SIZE
            for (t in 1 until SIZE - 1) {
                var c = 0
                for (db in -1..1) for (dt in -1..1) { c = (c shl 1) or (if (bits[b + db][t + dt]) 1 else 0) }
                codes[n++] = region * CODES + c
            }
        }
        return Encoded(img, codes, bandMean)
    }
}

class CountModel(val counts: IntArray = IntArray(Encoder.FEATURES), var segments: Int = 0) {
    private val regionTotals = LongArray(Encoder.REGIONS)
    // running statistics of the model's own score on data it was trained with
    var scoreMean = 0.0; var scoreM2 = 0.0; var scoreN = 0

    init { recomputeTotals() }

    private fun recomputeTotals() {
        regionTotals.fill(0)
        for (i in counts.indices) regionTotals[i / Encoder.CODES] += counts[i].toLong()
    }

    fun add(e: Encoder.Encoded) {
        if (segments >= 3) {   // track score distribution before learning the segment
            val s = score(e)
            scoreN++; val d = s - scoreMean; scoreMean += d / scoreN; scoreM2 += d * (s - scoreMean)
        }
        for (c in e.codes) { counts[c]++; regionTotals[c / Encoder.CODES]++ }
        segments++
    }

    /** mean log-probability of the segment's patterns under this model */
    fun score(e: Encoder.Encoded): Double {
        var s = 0.0
        for (c in e.codes) {
            val r = c / Encoder.CODES
            s += ln((counts[c] + 0.5) / (regionTotals[r] + 0.5 * Encoder.CODES))
        }
        return s / e.codes.size
    }

    val scoreStd: Double get() = if (scoreN > 1) sqrt(scoreM2 / (scoreN - 1)).coerceAtLeast(0.02) else 0.2

    /** how unusual a segment is: negative = less likely than usual */
    fun z(e: Encoder.Encoded): Double = if (scoreN < 3) 0.0 else (score(e) - scoreMean) / scoreStd

    fun copy(): CountModel = CountModel(counts.copyOf(), segments).also {
        it.scoreMean = scoreMean; it.scoreM2 = scoreM2; it.scoreN = scoreN
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("segments", segments); put("mean", scoreMean); put("m2", scoreM2); put("n", scoreN)
        // sparse: index,count pairs
        val arr = JSONArray()
        for (i in counts.indices) if (counts[i] != 0) { arr.put(i); arr.put(counts[i]) }
        put("counts", arr)
    }

    companion object {
        fun fromJson(o: JSONObject): CountModel {
            val c = IntArray(Encoder.FEATURES)
            val arr = o.getJSONArray("counts")
            var i = 0
            while (i + 1 < arr.length()) { c[arr.getInt(i)] = arr.getInt(i + 1); i += 2 }
            return CountModel(c, o.getInt("segments")).also {
                it.scoreMean = o.getDouble("mean"); it.scoreM2 = o.getDouble("m2"); it.scoreN = o.getInt("n")
            }
        }
    }
}

/** First-level categories. Safety-related ones are surfaced first when parked. */
enum class Category(val label: String, val safety: Boolean) {
    BRAKE("제동 계열", true),
    SUSPENSION("하체 계열", true),
    TIRE("타이어·노면", true),
    ENGINE("엔진·구동계", false),
    RATTLE("실내 덜컹거림", false),
    WIND("풍절음", false),
}

/** Driving states; the normal baseline is learned separately for each. */
enum class DriveState(val label: String) {
    IDLE("정차·공회전"), BRAKING("제동"), TURNING("큰 조향"),
    ACCEL("가속"), DECEL("감속"), CRUISE("정속"),
}
