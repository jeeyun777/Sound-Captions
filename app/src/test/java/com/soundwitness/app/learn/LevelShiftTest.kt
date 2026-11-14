package com.soundwitness.app.learn

import com.soundwitness.app.audio.SAMPLE_RATE
import org.junit.Test
import kotlin.math.sqrt
import kotlin.random.Random

/** Normal sound that only got louder (rough road, rain, window open): should NOT look anomalous. */
class LevelShiftTest {
    private fun cabin(s: Int, level: Float): ShortArray { val x = FloatArray(SAMPLE_RATE); SyntheticSounds.cabin(x, Random(s), level); return ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() } }
    private fun auc(n: List<Double>, c: List<Double>): Double { var w = 0.0; for (a in c) for (b in n) w += if (a > b) 1.0 else if (a == b) 0.5 else 0.0; return w / (n.size * c.size) }

    @Test
    fun louderNormal() {
        val cm = CountModel(); val mean = DoubleArray(Encoder.SIZE); val m2 = DoubleArray(Encoder.SIZE); var k = 0
        repeat(60) { val e = Encoder.encode(cabin(it, 0.08f)); cm.add(e); k++; for (b in 0 until Encoder.SIZE) { val d = e.bandMeanDb[b] - mean[b]; mean[b] += d / k; m2[b] += d * (e.bandMeanDb[b] - mean[b]) } }
        fun band(e: Encoder.Encoded) = (0 until Encoder.SIZE).sumOf { b -> val z = (e.bandMeanDb[b] - mean[b]) / sqrt(m2[b] / (k - 1)).coerceAtLeast(0.3); z * z } / Encoder.SIZE
        for (lv in listOf(0.12f, 0.16f, 0.24f)) {
            val n = (0 until 40).map { Encoder.encode(cabin(20_000 + it, 0.08f)) }
            val c = (0 until 40).map { Encoder.encode(cabin(30_000 + it, lv)) }
            println("louder normal x%.1f  counting AUC %.3f | band-stat AUC %.3f  (0.5 = not fooled, 1.0 = false alarm)".format(lv / 0.08f, auc(n.map { -cm.z(it) }, c.map { -cm.z(it) }), auc(n.map(::band), c.map(::band))))
        }
    }
}
