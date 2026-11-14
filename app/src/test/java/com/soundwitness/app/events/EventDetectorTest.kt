package com.soundwitness.app.events

import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.learn.Encoder
import com.soundwitness.app.vehicle.VehicleSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class EventDetectorTest {

    private fun detector() = EventDetector().apply { trainPrototypes() }

    @Test
    fun recallAndFalseAlarms() {
        val d = detector()
        println("--- per-window recall on unseen synthetic events in cabin noise ---")
        for (lvl in listOf(0.08f, 0.15f, 0.3f)) {
            val row = EventType.entries.joinToString(" | ") { t ->
                val hit = (0 until 40).count { d.classify(EventSounds.inCabin(t, 300_000 + it * 13, lvl)).type == t }
                "%s %.2f".format(t.name, hit / 40.0)
            }
            println("event level %.2f : %s".format(lvl, row))
        }
        println("--- false alarms on cabin noise only (per window) ---")
        for (cl in listOf(0.04f, 0.08f, 0.15f, 0.25f)) {
            val fa = (0 until 100).count { d.classify(EventSounds.cabinOnly(400_000 + it, cl)).type != null }
            println("cabin level %.2f : %d / 100".format(cl, fa))
        }
        val faMedia = (0 until 40).count { d.classify(EventSounds.inCabin(EventType.SIREN, 500_000 + it, 0.08f), mediaPlaying = true).type != null }
        println("weak siren while media plays, flagged: $faMedia / 40 (threshold raised)")
        val recall = (0 until 40).count { d.classify(EventSounds.inCabin(EventType.SIREN, 600_000 + it, 0.3f)).type == EventType.SIREN }
        assertTrue("siren recall $recall/40", recall >= 34)
        val fa = (0 until 100).count { d.classify(EventSounds.cabinOnly(700_000 + it, 0.08f)).type != null }
        assertTrue("false alarms $fa/100", fa <= 3)
    }

    @Test
    fun hornMeaningDependsOnVehicleState() {
        fun snap(v: Float, gear: String, steer: Float = 0f, signal: String = "OFF") =
            VehicleSnapshot(0, false, steer, v, false, 0f, false, false, false, gear, signal, false)
        assertEquals("뒤차 경적", Interpreter.horn(snap(0f, "GEAR_DRIVE")).first)
        assertEquals("옆 차로 주의", Interpreter.horn(snap(50f, "GEAR_DRIVE", 12f, "RIGHT")).second)
        assertEquals("멈추고 주변을 확인하세요", Interpreter.horn(snap(3f, "GEAR_REVERSE")).second)
        assertEquals("주변 차량을 확인하세요", Interpreter.horn(snap(60f, "GEAR_DRIVE")).second)
    }

    /** a passing siren + driver slows down → learned; same sound while media plays and no reaction → held */
    @Test
    fun selfLabelFromReaction() {
        val d = detector()
        fun run(react: Boolean, media: Boolean): LogEntry {
            val eng = EventEngine(d)
            val r = Random(if (react) 1 else 2)
            var t = 0L
            for (k in 0 until 24) {           // 12 s of half-second windows; siren from 1 s to 10 s
                val x = FloatArray(SAMPLE_RATE)
                com.soundwitness.app.learn.SyntheticSounds.cabin(x, r, 0.08f)
                val sec = k * 0.5f
                if (sec in 1f..10f) {
                    val g = if (sec < 6f) 0.1f + 0.08f * (sec - 1f) else 0.5f - 0.08f * (sec - 6f)
                    EventSounds.siren(x, r, g, pattern = 0, phase = sec / 4f)
                }
                val v = if (react && sec > 4f) 15f else 50f
                val s = VehicleSnapshot(t, react && sec > 4f, 0f, v, false, 0f, false, false, false, "GEAR_DRIVE", "OFF", false)
                eng.push(d.classify(EventSounds.toPcm(x), media), t, s, media, true)
                eng.tick(t, s); t += 500
            }
            val s = VehicleSnapshot(t, false, 0f, 10f, false, 0f, false, false, false, "GEAR_DRIVE", "OFF", false)
            repeat(30) { t += 500; eng.tick(t, s) }
            return eng.log.first { it.kind == "SIREN" }
        }
        val learned = run(react = true, media = false)
        val held = run(react = false, media = true)
        println("reacted: ${learned.label} (${learned.detail}; ${learned.reason})")
        println("no reaction, media: ${held.label} (${held.detail}; ${held.reason})")
        assertEquals("학습 반영", learned.label)
        assertEquals("보류", held.label)
    }

    /** poison: horns wrongly learned as sirens → self test sees horn recall drop → rolls back */
    @Test
    fun selfTestRollsBackBadLearning() {
        val d = detector()
        val before = d.selfTest()
        println("before: recall %.2f, false %d/%d".format(before.recallRate, before.falseAlarms, before.bgWindows))
        repeat(4) { day -> // learnEvent is capped per call budget; feed several batches
            val bad = (0 until 60).map { Encoder.encode(EventSounds.inCabin(EventType.HORN, 800_000 + day * 100 + it, 0.3f)) }
            d.learnEvent(EventType.SIREN, bad)
        }
        val after = d.selfTest()
        println("after poisoning: rolledBack=${after.rolledBack}, recall %.2f, false %d/%d".format(after.recallRate, after.falseAlarms, after.bgWindows))
        assertTrue(after.rolledBack)
        assertTrue(after.recallRate >= before.recallRate - 0.05)
    }
}
