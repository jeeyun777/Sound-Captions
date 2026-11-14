package com.soundwitness.app.learn

import android.content.Context
import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.vehicle.VehicleSnapshot
import org.json.JSONObject
import java.io.File
import kotlin.math.abs

/**
 * Learns this car's normal sound, per driving state, while it drives.
 *
 *  - live baseline : keeps learning from ordinary segments
 *  - fixed baseline: frozen copy taken once enough segments are learned (or reset
 *                    after the user marks a repair as done), so a slowly growing
 *                    fault is not absorbed into "normal"
 *
 * Two detectors per state, combined (see BandStat / Combined):
 *  - band baseline with the common level shift removed: catches energy changes
 *    in some bands (squeal, knock, rattle) and ignores a louder road / rain
 *  - counting model: time-frequency pattern; also used for the sound category
 * deviation = max(band z, -counting z), positive = unusual.
 *
 * A segment is learned as normal only when it does not look like speech and its
 * deviation is below the gate.
 */
class Learner(private val context: Context) {

    companion object {
        const val FREEZE_AFTER = 40        // segments per state before the fixed baseline is frozen (demo scale)
        const val LEARN_GATE = 2.0         // do not learn segments that already look unusual
        const val ALERT_DEV = 2.0          // sustained deviation threshold
        const val RECENT = 30              // recent segments per state kept for the alert decision
    }

    val live = HashMap<DriveState, CountModel>()
    val fixed = HashMap<DriveState, CountModel>()
    /** band baseline, learned until the fixed baseline is frozen, then kept */
    val band = HashMap<DriveState, BandStat>()
    val recentDev = HashMap<DriveState, ArrayDeque<Double>>()
    private val recentBand = HashMap<DriveState, ArrayDeque<FloatArray>>()
    /** per-state history of the band-change value, for the alert chart */
    val history = HashMap<DriveState, ArrayDeque<Float>>()
    val categories = HashMap<Category, CountModel>()
    var learnedSegments = 0; private set
    var skippedSpeech = 0; private set

    private val file = File(context.filesDir, "learner.json")

    // ---------- driving state ----------

    private var lastSpeed: Float? = null
    fun stateOf(s: VehicleSnapshot): DriveState {
        val v = s.speedKmh ?: 0f
        val prev = lastSpeed; lastSpeed = v
        return when {
            v < 2f && s.brake != true -> DriveState.IDLE
            s.brake == true -> DriveState.BRAKING
            abs(s.steeringDeg ?: 0f) >= 30f -> DriveState.TURNING
            prev != null && v - prev > 2f -> DriveState.ACCEL
            prev != null && prev - v > 2f -> DriveState.DECEL
            else -> DriveState.CRUISE
        }
    }

    // ---------- continuous learning ----------

    data class SegmentResult(val state: DriveState, val deviation: Double, val countZ: Double, val bandZ: Double, val learned: Boolean, val speech: Boolean)

    @Synchronized
    fun onSegment(pcm: ShortArray, snap: VehicleSnapshot): SegmentResult {
        val state = stateOf(snap)
        val frames = Dsp.bandFrames(pcm, 0, pcm.size, Encoder.edges)
        if (Dsp.looksLikeSpeech(frames, Encoder.edges)) { skippedSpeech++; return SegmentResult(state, 0.0, 0.0, 0.0, false, true) }
        val e = Encoder.encode(pcm)
        val model = live.getOrPut(state) { CountModel() }
        val ref = fixed[state] ?: model
        val bs = band.getOrPut(state) { BandStat() }
        val zc = ref.z(e)
        val zb = bs.z(e.bandMeanDb)
        val dev = Combined.deviation(zc, zb)
        recentDev.getOrPut(state) { ArrayDeque() }.apply { addLast(dev); while (size > RECENT) removeFirst() }
        recentBand.getOrPut(state) { ArrayDeque() }.apply { addLast(e.bandMeanDb); while (size > RECENT) removeFirst() }

        val learn = dev < LEARN_GATE
        if (learn) {
            model.add(e); learnedSegments++
            if (fixed[state] == null) {
                bs.add(e.bandMeanDb)
                if (model.segments >= FREEZE_AFTER) fixed[state] = model.copy()
            }
        }
        recordHistory(state)
        if (learnedSegments % 20 == 0) save()
        return SegmentResult(state, dev, zc, zb, learn, false)
    }

    private var histTick = 0
    private fun recordHistory(state: DriveState) {
        if (fixed[state] == null) return
        if (++histTick % 5 != 0) return
        val change = bandChange(state)?.second ?: return
        history.getOrPut(state) { ArrayDeque() }.apply { addLast(change); while (size > 28) removeFirst() }
    }

    /** band with the largest rise vs normal after the common level shift is removed, and its dB change */
    fun bandChange(state: DriveState): Pair<Int, Float>? {
        val bs = band[state]?.takeIf { it.k > 1 } ?: return null
        val recent = recentBand[state]?.takeIf { it.size >= 5 } ?: return null
        val avg = FloatArray(Encoder.SIZE) { b -> recent.map { it[b] }.average().toFloat() }
        val d = bs.shapeChange(avg)
        val best = d.indices.maxBy { d[it] }
        return best to d[best].toFloat()
    }

    /** common level change vs normal (dB), e.g. rough road or rain; shown, never alerted on */
    fun levelShift(state: DriveState): Float? {
        val bs = band[state]?.takeIf { it.k > 1 } ?: return null
        val recent = recentBand[state]?.takeIf { it.size >= 5 } ?: return null
        return bs.levelShift(FloatArray(Encoder.SIZE) { b -> recent.map { it[b] }.average().toFloat() }).toFloat()
    }

    data class Alert(val state: DriveState, val outOfRange: Int, val window: Int, val bandLowHz: Int, val bandHighHz: Int, val changeDb: Float)

    @Synchronized
    fun alerts(): List<Alert> = DriveState.entries.mapNotNull { st ->
        val zs = recentDev[st] ?: return@mapNotNull null
        if (fixed[st] == null || zs.size < 10) return@mapNotNull null
        val out = zs.count { it > ALERT_DEV }
        if (out < zs.size * 0.6) return@mapNotNull null
        val (b, d) = bandChange(st) ?: return@mapNotNull null
        val lo = Dsp.bandCenterHz(Encoder.edges, (b - 1).coerceAtLeast(0)).toInt()
        val hi = Dsp.bandCenterHz(Encoder.edges, (b + 1).coerceAtMost(Encoder.SIZE - 1)).toInt()
        Alert(st, out, zs.size, lo, hi, d)
    }

    /** user says the car was repaired: re-learn the fixed baseline from now on */
    @Synchronized
    fun resetAfterRepair() {
        fixed.clear(); band.clear(); recentDev.clear(); recentBand.clear(); history.clear()
        live.clear(); save()
    }

    // ---------- category model ----------

    fun trainCategories() {
        if (categories.isNotEmpty()) return
        for (c in Category.entries) {
            val m = CountModel()
            repeat(24) { i -> m.add(Encoder.encode(SyntheticSounds.sample(c, seed = i * 31 + c.ordinal))) }
            categories[c] = m
        }
    }

    fun classify(pcm: ShortArray, start: Int, length: Int): Pair<Category, Double> {
        trainCategories()
        val e = Encoder.encode(pcm, start, length)
        val scores = categories.mapValues { it.value.score(e) }
        val best = scores.maxBy { it.value }
        val second = scores.values.filter { it != best.value }.maxOrNull() ?: best.value
        return best.key to (best.value - second)
    }

    /** user-marked clip: learned as an example of its category only if the model agrees */
    fun learnUserClip(cat: Category, pcm: ShortArray, start: Int, length: Int, agreed: Boolean) {
        if (!agreed) return
        categories[cat]?.add(Encoder.encode(pcm, start, length))
    }

    // ---------- persistence ----------

    @Synchronized
    fun save() {
        val o = JSONObject()
        fun mapOut(m: Map<DriveState, CountModel>) = JSONObject().apply { m.forEach { (k, v) -> put(k.name, v.toJson()) } }
        o.put("live", mapOut(live)); o.put("fixed", mapOut(fixed))
        o.put("band", JSONObject().apply { band.forEach { (k, v) -> put(k.name, v.toJson()) } })
        o.put("learned", learnedSegments); o.put("speech", skippedSpeech)
        file.writeText(o.toString())
    }

    @Synchronized
    fun load() {
        if (!file.exists()) return
        runCatching {
            val o = JSONObject(file.readText())
            fun mapIn(j: JSONObject, into: HashMap<DriveState, CountModel>) = j.keys().forEach { into[DriveState.valueOf(it)] = CountModel.fromJson(j.getJSONObject(it)) }
            mapIn(o.getJSONObject("live"), live); mapIn(o.getJSONObject("fixed"), fixed)
            o.optJSONObject("band")?.let { bb -> bb.keys().forEach { k -> band[DriveState.valueOf(k)] = BandStat.fromJson(bb.getJSONObject(k)) } }
            learnedSegments = o.optInt("learned"); skippedSpeech = o.optInt("speech")
        }
    }

    fun totalLearnedFor(state: DriveState) = live[state]?.segments ?: 0
}
