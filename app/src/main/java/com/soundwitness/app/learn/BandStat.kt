package com.soundwitness.app.learn

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Simple per-band baseline (mean / spread of band dB) used together with the
 * counting model.
 *
 *  - band shape : deviation per band after removing the common level shift
 *                 (median dB change over all bands). A louder road, rain or an
 *                 open window raises every band together and is removed here;
 *                 a new squeal, knock or rattle raises only some bands.
 *  - counting   : time-frequency pattern (see CountModel), level-normalized.
 *
 * Combined deviation = max(band-shape z, -counting z). Either one can raise it,
 * neither reacts to an overall level change.
 */
class BandStat(
    val mean: DoubleArray = DoubleArray(Encoder.SIZE),
    val m2: DoubleArray = DoubleArray(Encoder.SIZE),
    var k: Int = 0,
) {
    companion object {
        const val SD_FLOOR_DB = 0.3
        const val SCORE_FROM = 10          // start score statistics once the band means settle

        fun fromJson(o: JSONObject): BandStat {
            val m = o.getJSONArray("mean"); val v = o.getJSONArray("m2")
            return BandStat(DoubleArray(m.length()) { m.getDouble(it) }, DoubleArray(v.length()) { v.getDouble(it) }, o.getInt("k")).apply {
                scoreN = o.optInt("sN"); scoreMean = o.optDouble("sMean", 0.0); scoreM2 = o.optDouble("sM2", 0.0)
            }
        }
    }

    var scoreN = 0; var scoreMean = 0.0; var scoreM2 = 0.0

    private fun sd(b: Int) = if (k > 1) sqrt(m2[b] / (k - 1)).coerceAtLeast(SD_FLOOR_DB) else 1.0

    /** common dB change of all bands vs normal (median, so one loud band does not move it) */
    fun levelShift(x: FloatArray): Double = DoubleArray(mean.size) { x[it] - mean[it] }.sorted().let { (it[it.size / 2] + it[(it.size - 1) / 2]) / 2 }

    /** per-band dB change after the common level shift is removed */
    fun shapeChange(x: FloatArray): DoubleArray { val sh = levelShift(x); return DoubleArray(mean.size) { x[it] - mean[it] - sh } }

    fun score(x: FloatArray): Double { val d = shapeChange(x); var s = 0.0; for (b in d.indices) { val z = d[b] / sd(b); s += z * z }; return s / d.size }

    fun add(x: FloatArray) {
        if (k >= SCORE_FROM) { val s = score(x); scoreN++; val d = s - scoreMean; scoreMean += d / scoreN; scoreM2 += d * (s - scoreMean) }
        k++
        for (b in mean.indices) { val d = x[b] - mean[b]; mean[b] += d / k; m2[b] += d * (x[b] - mean[b]) }
    }

    private val scoreStd get() = if (scoreN > 1) sqrt(scoreM2 / (scoreN - 1)).coerceAtLeast(0.05) else 1.0

    /** positive = band shape further from normal than usual */
    fun z(x: FloatArray): Double = if (scoreN < 3) 0.0 else (score(x) - scoreMean) / scoreStd

    fun copy() = BandStat(mean.copyOf(), m2.copyOf(), k).also { it.scoreN = scoreN; it.scoreMean = scoreMean; it.scoreM2 = scoreM2 }

    fun toJson(): JSONObject = JSONObject().apply {
        put("mean", JSONArray(mean.toList())); put("m2", JSONArray(m2.toList())); put("k", k)
        put("sN", scoreN); put("sMean", scoreMean); put("sM2", scoreM2)
    }
}

object Combined {
    /** positive = unusual. counting z is negative when unusual, band z positive. */
    fun deviation(countZ: Double, bandZ: Double): Double = max(-countZ, bandZ)
}
