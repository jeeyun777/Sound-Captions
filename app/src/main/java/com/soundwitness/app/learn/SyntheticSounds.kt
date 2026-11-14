package com.soundwitness.app.learn

import com.soundwitness.app.audio.SAMPLE_RATE
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

/**
 * Procedural prototypes of each noise category, used to give the category model
 * its starting counts before any real recordings exist. The same recipes are
 * used by tools/make_demo_sounds.py to create the sounds played in the demo.
 * These are stand-ins, not recordings of real faults.
 */
object SyntheticSounds {

    fun sample(cat: Category, seed: Int, seconds: Float = 1f, withCabin: Boolean = true): ShortArray {
        val r = Random(seed)
        val n = (SAMPLE_RATE * seconds).toInt()
        val x = FloatArray(n)
        if (withCabin) cabin(x, r, 0.08f)
        when (cat) {
            Category.BRAKE -> squeal(x, r)
            Category.SUSPENSION -> knocks(x, r)
            Category.TIRE -> tire(x, r)
            Category.ENGINE -> engine(x, r)
            Category.RATTLE -> rattle(x, r)
            Category.WIND -> wind(x, r)
        }
        return ShortArray(n) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }
    }

    /** normal cabin sound: low road rumble plus a weak engine hum */
    fun cabin(x: FloatArray, r: Random, level: Float) {
        var lp = 0f
        val hum = 28f + r.nextFloat() * 20f
        for (i in x.indices) {
            lp += 0.02f * ((r.nextFloat() * 2 - 1) - lp)
            val t = i.toFloat() / SAMPLE_RATE
            x[i] += level * (6f * lp + 0.25f * sin(2 * PI * hum * t).toFloat() + 0.1f * sin(4 * PI * hum * t).toFloat())
        }
    }

    private fun squeal(x: FloatArray, r: Random) {
        val f = 6000f + r.nextFloat() * 1800f
        val on = r.nextFloat() * 0.3f; val len = 0.4f + r.nextFloat() * 0.5f
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            if (t < on || t > on + len) continue
            val env = sin(PI * (t - on) / len).toFloat()
            val vib = 1f + 0.004f * sin(2 * PI * 6 * t).toFloat()
            x[i] += 0.35f * env * sin(2 * PI * f * vib * t).toFloat()
        }
    }

    private fun knocks(x: FloatArray, r: Random) {
        repeat(2 + r.nextInt(3)) {
            val at = (r.nextFloat() * 0.9f * x.size).toInt()
            val f = 80f + r.nextFloat() * 200f
            for (k in 0 until 2400) {
                val i = at + k; if (i >= x.size) break
                val t = k.toFloat() / SAMPLE_RATE
                x[i] += 0.6f * exp(-t * 35f) * sin(2 * PI * f * t).toFloat()
            }
        }
    }

    private fun tire(x: FloatArray, r: Random) {
        val rate = 5f + r.nextFloat() * 5f
        var bp1 = 0f; var bp2 = 0f
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            val w = r.nextFloat() * 2 - 1
            bp1 += 0.25f * (w - bp1); bp2 += 0.25f * (bp1 - bp2)
            val mod = 0.6f + 0.4f * sin(2 * PI * rate * t).toFloat()
            x[i] += 0.5f * mod * (bp1 - bp2)
        }
    }

    private fun engine(x: FloatArray, r: Random) {
        val f0 = 90f + r.nextFloat() * 60f
        val belt = 2600f + r.nextFloat() * 900f
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            var s = 0f
            for (h in 1..6) s += (0.25f / h) * sin(2 * PI * f0 * h * t).toFloat()
            val chirp = (sin(2 * PI * 3 * t).toFloat()).coerceAtLeast(0f)
            s += 0.18f * chirp * sin(2 * PI * belt * t).toFloat()
            x[i] += s
        }
    }

    private fun rattle(x: FloatArray, r: Random) {
        var t = 0
        while (t < x.size) {
            t += (SAMPLE_RATE / (8 + r.nextInt(14)))
            val f = 2500f + r.nextFloat() * 2500f
            for (k in 0 until 160) {
                val i = t + k; if (i >= x.size) break
                x[i] += 0.4f * exp(-k / 30f) * sin(2 * PI * f * k / SAMPLE_RATE).toFloat()
            }
        }
    }

    private fun wind(x: FloatArray, r: Random) {
        var prev = 0f; var slow = 0.5f
        for (i in x.indices) {
            val w = r.nextFloat() * 2 - 1
            val hp = w - prev; prev = w
            if (i % 400 == 0) slow = (slow + (r.nextFloat() - 0.5f) * 0.2f).coerceIn(0.3f, 1f)
            x[i] += 0.22f * slow * hp
        }
    }
}
