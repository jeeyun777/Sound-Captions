package com.soundwitness.app.learn

import com.soundwitness.app.audio.SAMPLE_RATE
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Counting model vs a simple band-statistics baseline (per-band mean/std of dB),
 * on synthetic cases. Reports AUC of separating normal from changed segments.
 */
class CompareBaselineTest {

    private fun toPcm(x: FloatArray) = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }

    /** normal: cabin + steady 7 kHz whine */
    private fun steadyTone(seed: Int): ShortArray {
        val r = Random(seed); val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, 0.08f)
        val f = 7000f + r.nextFloat() * 100f; val a = 0.05f * (0.9f + 0.2f * r.nextFloat())
        for (i in x.indices) x[i] += a * sin(2 * PI * f * i / SAMPLE_RATE).toFloat()
        return toPcm(x)
    }

    /** changed: same tone, same average power, but pulsing on/off (e.g. an intermittent squeal or rattle) */
    private fun pulsingTone(seed: Int): ShortArray {
        val r = Random(seed); val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, 0.08f)
        val f = 7000f + r.nextFloat() * 100f; val a = 0.05f * sqrt(2f) * (0.9f + 0.2f * r.nextFloat())
        val rate = 5f + r.nextFloat() * 3f; val ph = r.nextFloat()
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            val on = ((t * rate + ph) % 1f) < 0.5f
            if (on) x[i] += a * sin(2 * PI * f * t).toFloat()
        }
        return toPcm(x)
    }

    /** changed: louder squeal added on top (the easy case) */
    private fun addedSqueal(seed: Int) = SyntheticSounds.sample(Category.BRAKE, seed)

    private fun auc(normal: List<Double>, changed: List<Double>): Double {
        // higher score = more anomalous
        var win = 0.0
        for (c in changed) for (n in normal) win += if (c > n) 1.0 else if (c == n) 0.5 else 0.0
        return win / (normal.size * changed.size)
    }

    private class BandBaseline {
        val n = Encoder.SIZE; var k = 0
        val mean = DoubleArray(n); val m2 = DoubleArray(n)
        fun add(b: FloatArray) { k++; for (i in 0 until n) { val d = b[i] - mean[i]; mean[i] += d / k; m2[i] += d * (b[i] - mean[i]) } }
        fun score(b: FloatArray): Double { var s = 0.0; for (i in 0 until n) { val sd = sqrt(m2[i] / (k - 1)).coerceAtLeast(0.3); val z = (b[i] - mean[i]) / sd; s += z * z }; return s / n }
    }

    private fun run(name: String, normalGen: (Int) -> ShortArray, changedGen: (Int) -> ShortArray) {
        val cm = CountModel(); val bb = BandBaseline()
        repeat(60) { val e = Encoder.encode(normalGen(it)); cm.add(e); bb.add(e.bandMeanDb) }
        val testN = (0 until 40).map { Encoder.encode(normalGen(20_000 + it)) }
        val testC = (0 until 40).map { Encoder.encode(changedGen(30_000 + it)) }
        val aucCount = auc(testN.map { -cm.z(it) }, testC.map { -cm.z(it) })
        val aucBand = auc(testN.map { bb.score(it.bandMeanDb) }, testC.map { bb.score(it.bandMeanDb) })
        println("%-38s counting AUC %.3f | band-stat AUC %.3f".format(name, aucCount, aucBand))
    }

    @Test
    fun compare() {
        val cabin: (Int) -> ShortArray = { s -> val x = FloatArray(SAMPLE_RATE); SyntheticSounds.cabin(x, Random(s), 0.08f); toPcm(x) }
        run("louder squeal added to cabin sound", cabin, ::addedSqueal)
        run("steady whine -> pulsing, same power", ::steadyTone, ::pulsingTone)
        run("cabin -> rattle clicks added", cabin) { SyntheticSounds.sample(Category.RATTLE, it) }
        run("cabin -> knock added", cabin) { SyntheticSounds.sample(Category.SUSPENSION, it) }
    }
}
