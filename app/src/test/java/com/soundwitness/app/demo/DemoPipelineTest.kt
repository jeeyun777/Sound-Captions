package com.soundwitness.app.demo

import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.events.EventDetector
import com.soundwitness.app.events.EventEngine
import com.soundwitness.app.learn.Encoder
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * Runs the 80-second demo through the same path as Controller.onWindow
 * (1 s window every 0.5 s → speech filter → classify → engine), without Android,
 * and prints what the driving screen and the parked log would show.
 */
class DemoPipelineTest {

    @Test
    fun demoScenarioProducesExpectedAlertsAndLabels() {
        val det = EventDetector().apply { trainPrototypes() }
        det.selfTest()
        val eng = EventEngine(det)
        val demo = DemoScenario({}, playAudio = false)
        val r = Random(42)
        val chunk = SAMPLE_RATE / 10
        val last = ShortArray(SAMPLE_RATE); var lastPos = 0; var sinceHalf = 0; var filled = 0
        val t0 = 1_700_000_000_000L
        var n = 0L
        var shown = ""
        val speechDropped = HashMap<Int, Int>()
        val alertsSeen = LinkedHashSet<String>()
        while (n < (DemoScenario.LENGTH_S * SAMPLE_RATE).toLong()) {
            val pcm = demo.render(n, chunk, r)
            for (k in pcm.indices) {
                last[lastPos] = pcm[k]; lastPos = (lastPos + 1) % SAMPLE_RATE
                if (filled < SAMPLE_RATE) { filled++; continue }      // same warm-up as AudioRing
                if (++sinceHalf >= SAMPLE_RATE / 2) {
                    sinceHalf = 0
                    val win = ShortArray(SAMPLE_RATE) { last[(lastPos + it) % SAMPLE_RATE] }
                    val s = (n + k).toFloat() / SAMPLE_RATE
                    val now = t0 + (s * 1000).toLong()
                    val snap = demo.snapshotAt(s, now)
                    val media = demo.mediaAt(s)
                    val frames = Dsp.bandFrames(win, 0, win.size, Encoder.edges)
                    if (Dsp.looksLikeSpeech(frames, Encoder.edges)) { speechDropped.merge(s.toInt(), 1, Int::plus); continue }
                    val w = det.classify(win, media)
                    eng.push(w, now, snap, media, snap.rearOccupied != false)
                    if (w.type == null && !media && snap.gear != "GEAR_PARK") det.learnBackground(win, w)
                    eng.tick(now, snap)
                    val a = eng.alert
                    val key = a?.let { "${it.title} / ${it.sub}" + (if (it.trend != 0) " trend=${it.trend}" else "") + (if (it.chips.isNotEmpty()) " " + it.chips.joinToString { (k, v) -> "$k $v" } else "") } ?: "-"
                    if (key != shown) { println("%5.1fs  %s".format(s, key)); shown = key; if (a != null) alertsSeen += a.title }
                }
            }
            n += chunk
        }
        // parked: let pending decisions settle
        var now = t0 + 80_000
        repeat(40) { now += 500; eng.tick(now, demo.snapshotAt(80f, now)) }
        val test = det.selfTest()
        println("speech filter dropped (second → windows): $speechDropped")
        println("--- parked log ---")
        eng.log.forEach { println("%5.1fs  %-12s %-28s [%s] %s".format((it.timeMs - t0) / 1000f, it.title, it.detail, it.label, it.reason ?: "")) }
        println("self test: ${test.recall} false ${test.falseAlarms}/${test.bgWindows} rolledBack=${test.rolledBack}")

        val kinds = eng.log.map { it.kind }
        assertTrue("no alert in the first 3 s: ${eng.log.firstOrNull()}", eng.log.none { it.timeMs - t0 < 3000 })
        assertTrue("siren logged: $kinds", "SIREN" in kinds)
        assertTrue("horn logged: $kinds", kinds.count { it == "HORN" } >= 2)
        assertTrue("baby logged: $kinds", "BABY" in kinds)
        assertTrue("siren learned", eng.log.any { it.kind == "SIREN" && it.label == "학습 반영" })
    }
}
