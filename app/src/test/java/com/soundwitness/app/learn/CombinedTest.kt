package com.soundwitness.app.learn

import com.soundwitness.app.audio.SAMPLE_RATE
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Combined detector (band-shape baseline + counting) vs each one alone, on synthetic cases.
 * AUC: 1.0 = always separates changed from normal. For "louder normal" 0.5 is the goal.
 */
class CombinedTest {
    private fun toPcm(x: FloatArray) = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }
    private fun cabin(s: Int, level: Float = 0.08f): ShortArray { val x = FloatArray(SAMPLE_RATE); SyntheticSounds.cabin(x, Random(s), level); return toPcm(x) }
    private fun auc(n: List<Double>, c: List<Double>): Double { var w = 0.0; for (a in c) for (b in n) w += if (a > b) 1.0 else if (a == b) 0.5 else 0.0; return w / (n.size * c.size) }

    private fun tone(seed: Int, pulsing: Boolean): ShortArray {
        val r = Random(seed); val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, 0.08f)
        val f = 7000f + r.nextFloat() * 100f; val a = 0.05f * (if (pulsing) sqrt(2f) else 1f) * (0.9f + 0.2f * r.nextFloat())
        val rate = 5f + r.nextFloat() * 3f; val ph = r.nextFloat()
        for (i in x.indices) { val t = i.toFloat() / SAMPLE_RATE; if (!pulsing || ((t * rate + ph) % 1f) < 0.5f) x[i] += a * sin(2 * PI * f * t).toFloat() }
        return toPcm(x)
    }

    /** a rattle that clicks at a steady rate; the same clicks at another rate keep band energy equal */
    private fun rhythm(seed: Int, hz: Float): ShortArray {
        val r = Random(seed); val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, 0.08f)
        val period = SAMPLE_RATE / hz; val clickLen = 160; val ph = r.nextFloat() * period
        val amp = 0.25f / sqrt(hz / 4f)             // equal total energy per second
        var t = ph
        while (t < x.size) {
            val s = t.toInt()
            for (i in 0 until clickLen) if (s + i < x.size) x[s + i] += amp * (r.nextFloat() * 2 - 1) * (1f - i.toFloat() / clickLen) * sin(2 * PI * 3200 * i / SAMPLE_RATE).toFloat()
            t += period
        }
        return toPcm(x)
    }

    /** tone switching on/off, 50 % duty; only the rate changes */
    private fun pulse(seed: Int, hz: Float): ShortArray {
        val r = Random(seed); val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, 0.08f)
        val f = 2500f + r.nextFloat() * 100f; val ph = r.nextFloat()
        for (i in x.indices) { val t = i.toFloat() / SAMPLE_RATE; if (((t * hz + ph) % 1f) < 0.5f) x[i] += 0.07f * sin(2 * PI * f * t).toFloat() }
        return toPcm(x)
    }

    private class Scores(val count: List<Double>, val band: List<Double>, val comb: List<Double>)

    private fun train(normal: (Int) -> ShortArray): Pair<CountModel, BandStat> {
        val cm = CountModel(); val bs = BandStat()
        repeat(60) { val e = Encoder.encode(normal(it)); cm.add(e); bs.add(e.bandMeanDb) }
        return cm to bs
    }

    private fun scores(cm: CountModel, bs: BandStat, gen: (Int) -> ShortArray, from: Int): Scores {
        val es = (0 until 40).map { Encoder.encode(gen(from + it)) }
        val c = es.map { -cm.z(it) }; val b = es.map { bs.z(it.bandMeanDb) }
        return Scores(c, b, es.indices.map { Combined.deviation(-c[it], b[it]) })
    }

    private val results = mutableMapOf<String, Double>()

    private fun case(name: String, normal: (Int) -> ShortArray, changed: (Int) -> ShortArray) {
        val (cm, bs) = train(normal)
        val n = scores(cm, bs, normal, 20_000); val c = scores(cm, bs, changed, 30_000)
        val r = Triple(auc(n.count, c.count), auc(n.band, c.band), auc(n.comb, c.comb))
        results[name] = r.third
        println("%-30s counting %.3f | band-shape %.3f | combined %.3f".format(name, r.first, r.second, r.third))
    }

    @Test
    fun combined() {
        val cab: (Int) -> ShortArray = { cabin(it) }
        println("--- anomaly (1.0 = always detected) ---")
        case("squeal added", cab) { SyntheticSounds.sample(Category.BRAKE, it) }
        case("whine -> pulsing, same power", { tone(it, false) }, { tone(it, true) })
        case("rattle added", cab) { SyntheticSounds.sample(Category.RATTLE, it) }
        case("knock added", cab) { SyntheticSounds.sample(Category.SUSPENSION, it) }
        case("tire noise added", cab) { SyntheticSounds.sample(Category.TIRE, it) }
        case("engine noise added", cab) { SyntheticSounds.sample(Category.ENGINE, it) }
        case("click rhythm 4 Hz -> 11 Hz", { rhythm(it, 4f) }, { rhythm(it, 11f) })
        case("pulse rhythm 3 Hz -> 9 Hz", { pulse(it, 3f) }, { pulse(it, 9f) })
        println("--- louder normal (0.5 = not fooled, 1.0 = false alarm) ---")
        for (lv in listOf(0.12f, 0.16f, 0.24f)) case("normal x%.1f louder".format(lv / 0.08f), cab) { cabin(it, lv) }

        // false alarm rate on normal segments at the alert threshold
        val (cm, bs) = train(cab)
        val n = scores(cm, bs, cab, 40_000)
        for (thr in listOf(2.0, 2.5)) {
            fun rate(l: List<Double>) = l.count { it > thr }.toDouble() / l.size
            println("normal segments over %.1f : counting %.3f | band-shape %.3f | combined %.3f".format(thr, rate(n.count), rate(n.band), rate(n.comb)))
        }

        assertTrue(results.filterKeys { !it.startsWith("normal") }.values.all { it > 0.9 })
        assertTrue(results.filterKeys { it.startsWith("normal") }.values.all { it < 0.8 })
    }
}
