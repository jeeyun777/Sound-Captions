package com.soundwitness.app.learn

import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.audio.SAMPLE_RATE
import org.junit.Test
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/** (1) classification: counting vs band-mean nearest centroid
 *  (2) anomaly: counting with bits set relative to the car's normal band level */
class CompareVariantsTest {
    private fun toPcm(x: FloatArray) = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }
    private val cabin: (Int) -> ShortArray = { s -> val x = FloatArray(SAMPLE_RATE); SyntheticSounds.cabin(x, Random(s), 0.08f); toPcm(x) }

    @Test
    fun classification() {
        val train = Category.entries.associateWith { c -> (0 until 24).map { Encoder.encode(SyntheticSounds.sample(c, it * 31 + c.ordinal)) } }
        val counts = train.mapValues { (_, es) -> CountModel().also { m -> es.forEach(m::add) } }
        val centroids = train.mapValues { (_, es) -> FloatArray(Encoder.SIZE) { b -> es.map { it.bandMeanDb[b] }.average().toFloat() } }
        // harder test: quieter target sound under louder cabin noise
        for ((label, level) in listOf("normal mix" to 1f, "target at 40% level" to 0.4f, "target at 20% level" to 0.2f)) {
            var okC = 0; var okB = 0; var n = 0
            for (c in Category.entries) repeat(15) { i ->
                val clean = SyntheticSounds.sample(c, 10_000 + i * 7 + c.ordinal, withCabin = false)
                val bg = cabin(50_000 + i * 13 + c.ordinal)
                val mix = ShortArray(SAMPLE_RATE) { (bg[it] + clean[it] * level).toInt().coerceIn(-32768, 32767).toShort() }
                val e = Encoder.encode(mix)
                if (counts.maxBy { it.value.score(e) }.key == c) okC++
                if (centroids.minBy { (_, m) -> m.indices.sumOf { b -> ((m[b] - e.bandMeanDb[b]).toDouble()).let { it * it } } }.key == c) okB++
                n++
            }
            println("classification %-22s counting %.3f | band-centroid %.3f".format(label, okC.toDouble() / n, okB.toDouble() / n))
        }
    }

    /** counting where a cell is 1 when it is >= 3 dB above the normal level of that band */
    private class RelCount {
        val base = DoubleArray(Encoder.SIZE); var k = 0
        val counts = IntArray(Encoder.FEATURES); val tot = LongArray(Encoder.REGIONS)
        var sMean = 0.0; var sM2 = 0.0; var sN = 0
        fun codes(img: Array<FloatArray>): IntArray {
            val S = Encoder.SIZE
            val bits = Array(S) { b -> BooleanArray(S) { t -> img[b][t] >= base[b] + 3.0 } }
            val out = IntArray((S - 2) * (S - 2)); var n = 0
            for (b in 1 until S - 1) { val r = b * Encoder.REGIONS / S; for (t in 1 until S - 1) { var c = 0; for (db in -1..1) for (dt in -1..1) c = (c shl 1) or (if (bits[b + db][t + dt]) 1 else 0); out[n++] = r * Encoder.CODES + c } }
            return out
        }
        fun score(cs: IntArray) = cs.sumOf { c -> ln((counts[c] + 0.5) / (tot[c / Encoder.CODES] + 0.5 * Encoder.CODES)) } / cs.size
        fun calibrate(img: Array<FloatArray>) { k++; for (b in base.indices) base[b] += (img[b].average() - base[b]) / k }
        fun add(img: Array<FloatArray>) { val cs = codes(img); if (sN >= 0 && counts.sum() > 0) { val s = score(cs); sN++; val d = s - sMean; sMean += d / sN; sM2 += d * (s - sMean) }; cs.forEach { counts[it]++; tot[it / Encoder.CODES]++ } }
        fun z(img: Array<FloatArray>) = (score(codes(img)) - sMean) / sqrt(sM2 / (sN - 1)).coerceAtLeast(0.02)
    }

    private fun auc(n: List<Double>, c: List<Double>): Double { var w = 0.0; for (a in c) for (b in n) w += if (a > b) 1.0 else if (a == b) 0.5 else 0.0; return w / (n.size * c.size) }

    @Test
    fun relativeCountingAnomaly() {
        val cases = listOf<Pair<String, (Int) -> ShortArray>>(
            "louder squeal" to { s -> SyntheticSounds.sample(Category.BRAKE, s) },
            "knock" to { s -> SyntheticSounds.sample(Category.SUSPENSION, s) },
            "rattle" to { s -> SyntheticSounds.sample(Category.RATTLE, s) },
            "tire" to { s -> SyntheticSounds.sample(Category.TIRE, s) },
        )
        for ((name, gen) in cases) {
            val m = RelCount()
            val train = (0 until 60).map { Encoder.encode(cabin(it)).image }
            train.take(20).forEach(m::calibrate); train.forEach(m::add)
            val zn = (0 until 40).map { -m.z(Encoder.encode(cabin(20_000 + it)).image) }
            val zc = (0 until 40).map { -m.z(Encoder.encode(gen(30_000 + it)).image) }
            println("relative-bit counting AUC %-14s %.3f".format(name, auc(zn, zc)))
        }
    }
    @Suppress("unused") private val d = Dsp
}
