package com.soundwitness.app.events

import com.soundwitness.app.learn.Encoder
import com.soundwitness.app.vehicle.VehicleSnapshot
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

/** what the driving screen shows; one at a time */
data class Alert(
    val kind: String,            // SIREN, HORN, BABY, TURN
    val title: String,
    val sub: String,
    val trend: Int = 0,          // siren: +1 approaching, -1 leaving, 0 unknown
    val chips: List<Pair<String, String>> = emptyList(),
    val untilMs: Long,
)

/** one row of "오늘 들린 소리" */
data class LogEntry(
    val timeMs: Long, val kind: String, val title: String, var detail: String,
    var label: String = "판단 중", var reason: String = "",
) {
    fun toJson() = JSONObject().put("t", timeMs).put("k", kind).put("ti", title).put("d", detail).put("l", label).put("r", reason)
    companion object { fun fromJson(o: JSONObject) = LogEntry(o.getLong("t"), o.getString("k"), o.getString("ti"), o.getString("d"), o.optString("l"), o.optString("r")) }
}

/** what a horn means, given the car's state at that moment */
object Interpreter {
    fun horn(s: VehicleSnapshot): Pair<String, String> {
        val v = s.speedKmh ?: 0f
        val steer = abs(s.steeringDeg ?: 0f)
        val signal = s.turnSignal != null && s.turnSignal != "OFF"
        return when {
            s.gear == "GEAR_REVERSE" -> "경적" to "멈추고 주변을 확인하세요"
            v < 2f && s.gear != "GEAR_PARK" -> "뒤차 경적" to "정차 중 · 신호를 확인하세요"
            signal && steer >= 5f -> "경적" to "옆 차로 주의"
            else -> "경적" to "주변 차량을 확인하세요"
        }
    }

    fun chips(s: VehicleSnapshot): List<Pair<String, String>> = buildList {
        add("속도" to (s.speedKmh?.let { "%.0fkm/h".format(it) } ?: "-"))
        add("기어" to gearLabel(s.gear))
        add("브레이크" to when (s.brake) { true -> "밟음"; false -> "뗌"; null -> "-" })
        add("방향지시등" to when (s.turnSignal) { "LEFT" -> "좌측"; "RIGHT" -> "우측"; "OFF" -> "꺼짐"; else -> "-" })
    }

    fun gearLabel(g: String?) = when (g) { "GEAR_PARK" -> "P"; "GEAR_REVERSE" -> "R"; "GEAR_NEUTRAL" -> "N"; "GEAR_DRIVE" -> "D"; null -> "-"; else -> g.removePrefix("GEAR_").take(2) }
}

/**
 * Turns per-window results into events, alerts, log rows and self-labels.
 *
 * Self-labeling (no questions to the driver):
 *  - siren: learned only if the driver reacted (slowed / braked / moved right / stopped)
 *    AND the sound itself behaved like a passing siren (lasted, rose or fell in level)
 *    AND no media was playing. Otherwise held.
 *  - horn while stopped: learned if the car moved off within a few seconds.
 *  - horn during a lane change: learned if the lane change was abandoned (steering returned).
 *  - baby cry, turn signal: record only.
 */
class EventEngine(private val detector: EventDetector) {

    private class Active(val type: EventType, val start: Long, val snapAtStart: VehicleSnapshot, val log: LogEntry) {
        var last = start
        val levels = ArrayList<Pair<Long, Float>>()
        val encs = ArrayList<Encoder.Encoded>()
        var media = false
        var levelRange = 0f
    }

    private class Pending(val a: Active, val decideAt: Long, val snaps: ArrayList<VehicleSnapshot> = ArrayList())

    private val recent = HashMap<EventType, ArrayDeque<Boolean>>()
    private val active = HashMap<EventType, Active>()
    private val pending = ArrayList<Pending>()
    val log = ArrayList<LogEntry>()
    @Volatile var alert: Alert? = null; private set
    var onLearned: (() -> Unit)? = null

    private var signalOnSince = 0L; private var signalNotified = false

    private fun hits(t: EventType, need: Int, of: Int) = (recent[t] ?: ArrayDeque()).takeLast(of).count { it } >= need

    @Synchronized
    fun push(w: EventDetector.Window, now: Long, snap: VehicleSnapshot, media: Boolean, rearAllowed: Boolean) {
        for (t in EventType.entries) recent.getOrPut(t) { ArrayDeque() }.apply { addLast(w.type == t); while (size > 6) removeFirst() }
        val t = w.type
        if (t != null) {
            val firing = when (t) {
                EventType.SIREN -> hits(t, 2, 3)
                EventType.HORN -> w.margin > (EventDetector.THRESHOLD[t]!! + 0.08) || hits(t, 2, 3)
                EventType.BABY -> rearAllowed && hits(t, 2, 4)
            }
            val a = active[t]
            if (a != null) { a.last = now; a.levels += now to w.levelDb; a.encs += w.enc; a.media = a.media || media }
            else if (firing) startEvent(t, now, snap, w, media)
        }
        endStale(now)
        refreshAlert(now, snap)
    }

    /** end events not heard for 3 s */
    private fun endStale(now: Long) {
        for ((type, a) in active.entries.toList()) if (now - a.last > 3000) { active.remove(type); endEvent(a, now) }
    }

    private fun startEvent(t: EventType, now: Long, snap: VehicleSnapshot, w: EventDetector.Window, media: Boolean) {
        val entry = when (t) {
            EventType.SIREN -> LogEntry(now, "SIREN", "긴급차량 사이렌", "주행 중")
            EventType.HORN -> { val (ti, sub) = Interpreter.horn(snap); LogEntry(now, "HORN", if (sub.startsWith("옆")) "경적 · 옆 차로 주의" else ti, hornDetail(snap)) }
            EventType.BABY -> LogEntry(now, "BABY", "뒷좌석 아이 울음", "뒷좌석 착석 감지")
        }
        log += entry
        val a = Active(t, now, snap, entry).apply { levels += now to w.levelDb; encs += w.enc; this.media = media }
        active[t] = a
        when (t) {
            EventType.SIREN -> alert = Alert("SIREN", "긴급차량 접근 중", "우측으로 양보할 준비를 하세요", 0, untilMs = now + 4000)
            EventType.HORN -> { val (ti, sub) = Interpreter.horn(snap); alert = Alert("HORN", ti, sub, chips = Interpreter.chips(snap), untilMs = now + 5000) }
            EventType.BABY -> if (alert?.kind != "SIREN") alert = Alert("BABY", "뒷좌석 아이 울음", "안전한 곳에 정차해 살펴보세요", untilMs = now + 5000)
        }
    }

    private fun hornDetail(s: VehicleSnapshot): String = buildList {
        if ((s.speedKmh ?: 0f) < 2f) add("정차 중")
        add("${Interpreter.gearLabel(s.gear)}단")
        if (s.brake == true) add("브레이크 밟음")
        when (s.turnSignal) { "LEFT" -> add("좌측 방향지시등"); "RIGHT" -> add("우측 방향지시등") }
        s.steeringDeg?.let { if (abs(it) >= 5f) add("조향 %.0f°".format(abs(it))) }
    }.joinToString(" · ")

    /** siren level trend over the last 3 s, dB per second */
    private fun slope(a: Active): Float {
        val pts = a.levels.filter { it.first >= a.last - 3000 }
        if (pts.size < 3) return 0f
        val t0 = pts.first().first
        val xs = pts.map { (it.first - t0) / 1000f }; val ys = pts.map { it.second }
        val mx = xs.average().toFloat(); val my = ys.average().toFloat()
        var num = 0f; var den = 0f
        for (i in xs.indices) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) * (xs[i] - mx) }
        return if (den > 0) num / den else 0f
    }

    private fun refreshAlert(now: Long, snap: VehicleSnapshot) {
        active[EventType.SIREN]?.let { a ->
            val s = slope(a)
            val trend = if (s > 0.8f) 1 else if (s < -0.8f) -1 else alert?.trend ?: 0
            alert = if (trend < 0) Alert("SIREN", "긴급차량 멀어짐", "소리가 작아지는 중", -1, untilMs = now + 3000)
                    else Alert("SIREN", "긴급차량 접근 중", "우측으로 양보할 준비를 하세요", trend, untilMs = now + 3000)
            return
        }
        // turn signal left on while going straight
        val on = snap.turnSignal == "LEFT" || snap.turnSignal == "RIGHT"
        if (!on) { signalOnSince = 0; signalNotified = false }
        else {
            if (signalOnSince == 0L || abs(snap.steeringDeg ?: 0f) >= 8f) { signalOnSince = now; signalNotified = false }
            val straightFor = (now - signalOnSince) / 1000
            if (!signalNotified && straightFor >= 20 && (snap.speedKmh ?: 0f) > 15f && (alert == null || now > alert!!.untilMs)) {
                signalNotified = true
                val side = if (snap.turnSignal == "LEFT") "좌측" else "우측"
                alert = Alert("TURN", "방향지시등 켜져 있음", "$side · 직진 상태로 ${straightFor}초", untilMs = now + 5000)
                log += LogEntry(now, "TURN", "방향지시등 켜진 채 주행", "$side · 직진 상태로 ${straightFor}초", "기록만", "차량 신호로 판단")
            }
        }
        if (alert != null && now > alert!!.untilMs) alert = null
    }

    private fun endEvent(a: Active, now: Long) {
        val dur = (a.last - a.start) / 1000f
        if (a.type == EventType.SIREN) sirenSummary(a, dur)
        if (a.type == EventType.BABY) { a.log.label = "기록만"; a.log.reason = "${dur.toInt() + 1}초 지속 · 반응으로 판단하지 않음"; return }
        pending += Pending(a, now + 8000)
    }

    private fun sirenSummary(a: Active, dur: Float) {
        val lv = a.levels.map { it.second }
        val peak = a.levels.maxBy { it.second }.first
        val shape = when {
            peak > a.start + 1000 && peak < a.last - 1000 -> "다가옴 → 멀어짐"
            peak >= a.last - 1000 -> "다가옴"
            else -> "멀어짐"
        }
        a.log.detail = "$shape · ${"%.0f".format(dur + 1)}초" + if (a.media) " · 미디어 재생 중" else ""
        a.levelRange = (lv.maxOrNull() ?: 0f) - (lv.minOrNull() ?: 0f)
    }

    /** called every tick with the latest vehicle state; decides pending self-labels */
    @Synchronized
    fun tick(now: Long, snap: VehicleSnapshot) {
        endStale(now)
        for (p in pending) p.snaps += snap
        val done = pending.filter { now >= it.decideAt }
        for (p in done) { pending.remove(p); decide(p) }
        refreshAlert(now, snap)
    }

    private fun decide(p: Pending) {
        val a = p.a
        val e = a.log
        val s0 = a.snapAtStart
        val v0 = s0.speedKmh ?: 0f
        when (a.type) {
            EventType.SIREN -> {
                val slowed = p.snaps.any { (it.speedKmh ?: v0) <= v0 * 0.7f } && v0 > 10f
                val stopped = p.snaps.any { (it.speedKmh ?: 99f) < 3f }
                val braked = p.snaps.any { it.brake == true }
                val right = p.snaps.any { (it.steeringDeg ?: 0f) >= 10f }
                val reacted = slowed || stopped || braked || right
                val lasted = (a.last - a.start) >= 2500
                val moved = a.levelRange >= 3f
                val ok = reacted && lasted && moved && !a.media
                if (ok) {
                    detector.learnEvent(EventType.SIREN, a.encs)
                    e.label = "학습 반영"
                    e.reason = buildList { if (braked) add("제동"); if (slowed) add("감속"); if (right) add("우측 조향"); if (stopped) add("정차") }.joinToString(" · ") + " 반응"
                    onLearned?.invoke()
                } else {
                    e.label = "보류"
                    e.reason = buildList {
                        if (!reacted) add("운전자 반응 없음")
                        if (!lasted || !moved) add("소리 흐름 불일치")
                        if (a.media) add("미디어 재생 중")
                    }.joinToString(" · ")
                }
            }
            EventType.HORN -> {
                val wasStopped = v0 < 2f
                val movedOff = wasStopped && p.snaps.any { (it.speedKmh ?: 0f) > 5f }
                val laneChange = s0.turnSignal == "LEFT" || s0.turnSignal == "RIGHT"
                val aborted = laneChange && p.snaps.any { abs(it.steeringDeg ?: 0f) <= abs(s0.steeringDeg ?: 0f) - 5f }
                if (movedOff || aborted) {
                    detector.learnEvent(EventType.HORN, a.encs)
                    e.label = "학습 반영"; e.reason = if (movedOff) "알림 뒤 출발" else "차로 변경 중단 · 조향 복귀"
                    onLearned?.invoke()
                } else { e.label = "보류"; e.reason = "운전자 반응으로 확인 안 됨" }
            }
            EventType.BABY -> {}
        }
    }

    // ---------- persistence of today's log ----------
    @Synchronized fun logJson() = JSONArray(log.map { it.toJson() })
    @Synchronized fun loadLog(a: JSONArray) {
        val today = java.util.Calendar.getInstance().apply { set(java.util.Calendar.HOUR_OF_DAY, 0); set(java.util.Calendar.MINUTE, 0) }.timeInMillis
        log.clear(); for (i in 0 until a.length()) LogEntry.fromJson(a.getJSONObject(i)).takeIf { it.timeMs >= today }?.let { log += it }
    }
}
