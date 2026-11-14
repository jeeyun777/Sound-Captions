package com.soundwitness.app.events

import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.learn.CountModel
import com.soundwitness.app.learn.Encoder
import org.json.JSONObject
import java.util.EnumMap

enum class EventType(val label: String) { SIREN("긴급차량 사이렌"), HORN("경적"), BABY("뒷좌석 아이 울음") }

/**
 * Per-window event recognition with counting models.
 *
 *  - one counting model per event type, started from synthetic prototypes
 *  - one background model = this car's normal cabin sound. It adapts on its own
 *    (no labels needed) from windows that look like nothing but background.
 *  - a window is an event when the best event model beats the background model
 *    by a margin (log-likelihood per code). The margin is raised while media plays.
 *
 * Adaptation is guarded: daily caps, a frozen factory copy, and a self test that
 * rolls back to the last good models when recall drops or false alarms rise.
 */
class EventDetector {

    companion object {
        val THRESHOLD = mapOf(EventType.SIREN to 0.10, EventType.HORN to 0.12, EventType.BABY to 0.12)
        const val MEDIA_EXTRA = 0.10
        const val BG_DAILY_CAP = 1200          // ~10 min of half-second windows per day
        const val EVENT_DAILY_CAP = 60
        const val PROTO_PER_TYPE = 36
        const val TEST_PER_TYPE = 10
        const val BG_RING = 120
    }

    data class Window(val type: EventType?, val margin: Double, val levelDb: Float, val enc: Encoder.Encoded, val best: EventType)

    /** decision margin per type; starts from the defaults, replaced by the factory model's calibrated values */
    val thresholds: EnumMap<EventType, Double> = EnumMap<EventType, Double>(EventType::class.java).apply { putAll(THRESHOLD) }

    /** clean 1 s event sounds for the self test; synthetic unless real factory test sounds are installed */
    var testEvents: Map<EventType, List<FloatArray>>? = null

    /** this car's usual background level per band (dB); null = plain (absolute) encoding */
    var profile: FloatArray? = null; private set
    private var profileN = 0
    var useProfile = false

    var models: EnumMap<EventType, CountModel> = EnumMap(EventType::class.java); private set
    var background = CountModel(); private set
    /** "other sounds" (animals, alarms, voices, tools...): an event must beat this too */
    var other: CountModel? = null; private set
    private var factoryThr: Map<EventType, Double> = THRESHOLD
    private var factory: Pair<EnumMap<EventType, CountModel>, CountModel>? = null
    private var good: Pair<EnumMap<EventType, CountModel>, CountModel>? = null

    var bgAdapted = 0; private set
    var eventAdapted = 0; private set
    private var dayKey = ""
    private var bgToday = 0; private var evToday = 0

    /** recent background windows, kept in memory only, for the self test */
    private val bgRing = ArrayDeque<ShortArray>()

    // ---------- setup ----------

    fun trainPrototypes() {
        if (models.isNotEmpty()) return
        for (t in EventType.entries) {
            val m = CountModel()
            repeat(PROTO_PER_TYPE) { i ->
                val lvl = listOf(0.12f, 0.2f, 0.35f, 0.6f)[i % 4]
                m.add(Encoder.encode(EventSounds.inCabin(t, 1000 + i * 17 + t.ordinal * 7919, lvl)))
            }
            models[t] = m
        }
        val bg = CountModel()
        repeat(PROTO_PER_TYPE) { i -> bg.add(Encoder.encode(EventSounds.cabinOnly(50_000 + i, listOf(0.04f, 0.08f, 0.14f, 0.22f)[i % 4]))) }
        background = bg
        factory = snapshot(); good = snapshot()
    }

    /** install a factory model built elsewhere (e.g. from real recordings) */
    @Synchronized
    fun installFactory(m: Map<EventType, CountModel>, bg: CountModel, thr: Map<EventType, Double>? = null, prof: FloatArray? = null, oth: CountModel? = null) {
        if (oth != null) other = oth.copy()
        thr?.let { factoryThr = it.toMap() }
        if (prof != null) { profile = prof.copyOf(); profileN = 20; useProfile = true }
        models = EnumMap<EventType, CountModel>(EventType::class.java).apply { m.forEach { (k, v) -> put(k, v.copy()) } }
        background = bg.copy()
        thr?.let { thresholds.putAll(it) }
        factory = snapshot(); good = snapshot(); goodRecall = -1.0
    }

    private fun snapshot(): Pair<EnumMap<EventType, CountModel>, CountModel> =
        EnumMap<EventType, CountModel>(EventType::class.java).apply { models.forEach { (k, v) -> put(k, v.copy()) } } to background.copy()

    private fun restore(s: Pair<EnumMap<EventType, CountModel>, CountModel>) {
        models = EnumMap<EventType, CountModel>(EventType::class.java).apply { s.first.forEach { (k, v) -> put(k, v.copy()) } }
        background = s.second.copy()
    }

    // ---------- recognition ----------

    @Synchronized
    fun classify(pcm: ShortArray, mediaPlaying: Boolean = false): Window {
        val e = Encoder.encode(pcm, 0, minOf(pcm.size, SAMPLE_RATE), if (useProfile) profile else null)
        val sb = maxOf(background.score(e), other?.score(e) ?: Double.NEGATIVE_INFINITY)
        var best = EventType.SIREN; var bestMargin = Double.NEGATIVE_INFINITY
        for ((t, m) in models) { val d = m.score(e) - sb; if (d > bestMargin) { bestMargin = d; best = t } }
        val thr = (thresholds[best] ?: 0.1) + if (mediaPlaying) MEDIA_EXTRA else 0.0
        val type = if (bestMargin > thr) best else null
        return Window(type, bestMargin, Dsp.rmsDb(pcm, 0, pcm.size), e, best)
    }

    // ---------- adaptation ----------

    private fun rollDay() {
        val k = java.text.SimpleDateFormat("yyyyMMdd", java.util.Locale.US).format(java.util.Date())
        if (k != dayKey) { dayKey = k; bgToday = 0; evToday = 0 }
    }

    /** a window that is nothing but this car's background (no event, no speech, no media) */
    @Synchronized
    fun learnBackground(pcm: ShortArray, w: Window) {
        rollDay()
        bgRing.addLast(pcm.copyOf()); while (bgRing.size > BG_RING) bgRing.removeFirst()
        if (bgToday >= BG_DAILY_CAP || w.margin > 0) return
        if (useProfile) profile?.let { p ->          // follow this car's background level, slowly
            profileN = minOf(profileN + 1, 200)
            val m = w.enc.bandMeanDb
            for (b in p.indices) p[b] += (m[b] - p[b]) / profileN
        }
        background.add(w.enc); bgToday++; bgAdapted++
    }

    /** confirmed by the driver's reaction and the sound's own trajectory */
    @Synchronized
    fun learnEvent(type: EventType, encs: List<Encoder.Encoded>) {
        rollDay()
        for (e in encs) { if (evToday >= EVENT_DAILY_CAP) break; models[type]?.add(e); evToday++; eventAdapted++ }
    }

    /**
     * Thresholds from this car's own usual sound: the 99th percentile of each type's margin over
     * recent background windows (no labels needed), never below the factory value.
     */
    @Synchronized
    fun recalibrate() {
        if (bgRing.size < 20) return
        val encs = bgRing.map { Encoder.encode(it, 0, minOf(it.size, SAMPLE_RATE), if (useProfile) profile else null) }
        for (t in EventType.entries) {
            val m = models[t] ?: continue
            val ms = encs.map { e -> m.score(e) - maxOf(background.score(e), other?.score(e) ?: Double.NEGATIVE_INFINITY) }.sorted()
            val q = ms[(ms.size * 0.99).toInt().coerceAtMost(ms.size - 1)]
            thresholds[t] = maxOf(factoryThr[t] ?: 0.0, q + 0.01)
        }
    }

    // ---------- self test and rollback ----------

    data class TestResult(val recall: Map<EventType, Int>, val perType: Int, val falseAlarms: Int, val bgWindows: Int, val rolledBack: Boolean, val timeMs: Long) {
        val recallRate get() = recall.values.sum().toDouble() / (perType * recall.size).coerceAtLeast(1)
        val falseRate get() = falseAlarms.toDouble() / bgWindows.coerceAtLeast(1)
    }
    var lastTest: TestResult? = null; private set
    private var goodRecall = -1.0; private var goodFalse = 1.0
    private var goodPerType: Map<EventType, Int> = emptyMap()

    private fun backgrounds(): List<ShortArray> =
        if (bgRing.size >= 10) bgRing.toList() else List(20) { EventSounds.cabinOnly(90_000 + it) }

    private fun runTest(): TestResult {
        val bgs = backgrounds()
        val recall = EnumMap<EventType, Int>(EventType::class.java)
        for (t in EventType.entries) {
            var hit = 0
            val real = testEvents?.get(t)
            repeat(TEST_PER_TYPE) { i ->
                val ev = real?.let { it[i % it.size] } ?: EventSounds.clean(t, 777_000 + i * 31 + t.ordinal * 101, 1f, listOf(0.15f, 0.25f, 0.4f)[i % 3])
                val bg = bgs[i % bgs.size]
                val mix = ShortArray(SAMPLE_RATE) { k -> (bg.getOrElse(k) { 0 } + ev[k] * 32000).toInt().coerceIn(-32768, 32767).toShort() }
                if (classify(mix).type == t) hit++
            }
            recall[t] = hit
        }
        val fa = bgs.count { classify(it).type != null }
        return TestResult(recall, TEST_PER_TYPE, fa, bgs.size, false, System.currentTimeMillis())
    }

    /** run after a learning session (e.g. on parking). Keeps or rolls back what was learned. */
    @Synchronized
    fun selfTest(): TestResult {
        var r = runTest()
        val typeDrop = r.recall.any { (t, hit) -> hit <= (goodPerType[t] ?: 0) - 2 }
        if (goodRecall >= 0 && (typeDrop || r.recallRate < goodRecall - 0.05 || r.falseRate > goodFalse + 0.05)) {
            good?.let(::restore)
            r = runTest().copy(rolledBack = true)
        } else {
            good = snapshot(); goodRecall = r.recallRate; goodFalse = r.falseRate; goodPerType = r.recall.toMap()
        }
        lastTest = r
        return r
    }

    @Synchronized
    fun resetToFactory() { factory?.let(::restore); good = snapshot(); goodRecall = -1.0 }

    // ---------- persistence ----------

    @Synchronized
    fun toJson(): JSONObject = JSONObject().apply {
        put("models", JSONObject().apply { models.forEach { (k, v) -> put(k.name, v.toJson()) } })
        put("background", background.toJson())
        put("bgAdapted", bgAdapted); put("eventAdapted", eventAdapted)
        put("goodRecall", goodRecall); put("goodFalse", goodFalse)
        put("thresholds", JSONObject().apply { thresholds.forEach { (k, v) -> put(k.name, v) } })
    }

    @Synchronized
    fun load(o: JSONObject) {
        val ms = o.getJSONObject("models")
        val loaded = EnumMap<EventType, CountModel>(EventType::class.java)
        ms.keys().forEach { loaded[EventType.valueOf(it)] = CountModel.fromJson(ms.getJSONObject(it)) }
        if (loaded.size == EventType.entries.size) models = loaded
        background = CountModel.fromJson(o.getJSONObject("background"))
        bgAdapted = o.optInt("bgAdapted"); eventAdapted = o.optInt("eventAdapted")
        goodRecall = o.optDouble("goodRecall", -1.0); goodFalse = o.optDouble("goodFalse", 1.0)
        o.optJSONObject("thresholds")?.let { t -> EventType.entries.forEach { e -> if (t.has(e.name)) thresholds[e] = t.getDouble(e.name) } }
        good = snapshot()
    }
}
