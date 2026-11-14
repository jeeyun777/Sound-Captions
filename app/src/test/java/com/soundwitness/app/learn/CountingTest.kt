package com.soundwitness.app.learn

import com.soundwitness.app.audio.SAMPLE_RATE
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class CountingTest {

    private fun cabinOnly(seed: Int): ShortArray {
        val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, Random(seed), 0.08f)
        return ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }
    }

    @Test
    fun categoriesAreSeparable() {
        val models = Category.entries.associateWith { c ->
            CountModel().also { m -> repeat(24) { i -> m.add(Encoder.encode(SyntheticSounds.sample(c, seed = i * 31 + c.ordinal))) } }
        }
        var correct = 0; var total = 0
        val confusion = StringBuilder()
        for (c in Category.entries) {
            val row = IntArray(Category.entries.size)
            repeat(15) { i ->
                val e = Encoder.encode(SyntheticSounds.sample(c, seed = 10_000 + i * 7 + c.ordinal))
                val pred = models.maxBy { it.value.score(e) }.key
                row[pred.ordinal]++
                if (pred == c) correct++
                total++
            }
            confusion.appendLine("${c.name.padEnd(10)} ${row.joinToString(" ") { it.toString().padStart(2) }}")
        }
        println(confusion)
        val acc = correct.toDouble() / total
        println("category accuracy on unseen synthetic samples: %.3f".format(acc))
        assertTrue(acc > 0.8)
    }

    @Test
    fun baselineFlagsAddedSqueal() {
        val normal = CountModel()
        repeat(60) { normal.add(Encoder.encode(cabinOnly(it))) }
        val zNormal = (0 until 20).map { normal.z(Encoder.encode(cabinOnly(5000 + it))) }
        val zSqueal = (0 until 20).map { normal.z(Encoder.encode(SyntheticSounds.sample(Category.BRAKE, seed = 7000 + it))) }
        println("z normal mean %.2f / squeal mean %.2f".format(zNormal.average(), zSqueal.average()))
        assertTrue(zSqueal.average() < -2.0)
        assertTrue(zNormal.average() > -1.5)
    }
}
