package com.soundwitness.app.events

import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.learn.SyntheticSounds
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

/**
 * Procedural stand-ins for the event sounds. They give the event models their
 * starting counts and drive the demo mode; real evaluation uses public datasets
 * (UrbanSound8K, ESC-50) mixed with recorded cabin noise.
 *
 * Korean emergency sirens use the common wail / yelp / hi-lo patterns; horns are
 * two steady tones with harmonics; a baby cry is a harmonic voice with a
 * rising-falling pitch in bursts.
 */
object EventSounds {

    /** clean event of [seconds], amplitude about [level] (no cabin noise) */
    fun clean(type: EventType, seed: Int, seconds: Float = 1f, level: Float = 0.3f): FloatArray {
        val r = Random(seed)
        val x = FloatArray((SAMPLE_RATE * seconds).toInt())
        when (type) {
            EventType.SIREN -> siren(x, r, level, pattern = r.nextInt(3), phase = r.nextFloat())
            EventType.HORN -> horn(x, r, level)
            EventType.BABY -> baby(x, r, level)
        }
        return x
    }

    /** event mixed into normal cabin sound, as heard inside the car */
    fun inCabin(type: EventType, seed: Int, eventLevel: Float, cabinLevel: Float = 0.08f): ShortArray {
        val r = Random(seed * 7 + 3)
        val x = FloatArray(SAMPLE_RATE)
        SyntheticSounds.cabin(x, r, cabinLevel)
        val e = clean(type, seed, 1f, eventLevel)
        for (i in x.indices) x[i] += e[i]
        return toPcm(x)
    }

    fun cabinOnly(seed: Int, level: Float = 0.08f): ShortArray {
        val x = FloatArray(SAMPLE_RATE); SyntheticSounds.cabin(x, Random(seed), level); return toPcm(x)
    }

    fun toPcm(x: FloatArray) = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }

    /**
     * Siren with a slowly changing level and pitch (approaching / passing / leaving).
     * [gain] gives the level per sample (0..1), [doppler] the pitch factor per sample.
     */
    fun siren(x: FloatArray, r: Random, level: Float, pattern: Int, phase: Float,
              gain: (Int) -> Float = { 1f }, doppler: (Int) -> Float = { 1f }) {
        var ph = 0.0
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE + phase * 4f
            val base = when (pattern) {
                0 -> 650f + 450f * (0.5f - 0.5f * kotlin.math.cos(2 * PI * t / 4.0).toFloat())    // wail, 4 s sweep
                1 -> 700f + 600f * ((t * 3.5f) % 1f)                                                // yelp, fast sweep
                else -> if (((t / 0.6f).toInt() % 2) == 0) 960f else 770f                           // hi-lo
            }
            val f = base * doppler(i)
            ph += 2 * PI * f / SAMPLE_RATE
            // square-ish tone: fundamental + odd harmonics (sirens are not pure tones)
            val s = sin(ph) + 0.33 * sin(3 * ph) + 0.2 * sin(5 * ph)
            x[i] += (level * gain(i) * 0.7 * s).toFloat()
        }
    }

    private fun horn(x: FloatArray, r: Random, level: Float) {
        val f1 = 400f + r.nextFloat() * 80f; val f2 = f1 * (1.24f + r.nextFloat() * 0.05f)
        val on = r.nextFloat() * 0.2f; val len = 0.35f + r.nextFloat() * 0.6f
        val twice = r.nextBoolean()
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            val inFirst = t >= on && t < on + len
            val inSecond = twice && t >= on + len + 0.12f && t < on + 2 * len + 0.12f
            if (!inFirst && !inSecond) continue
            var s = 0.0
            for (h in 1..6) s += (1.0 / h) * (sin(2 * PI * f1 * h * t) + sin(2 * PI * f2 * h * t))
            x[i] += (level * 0.45 * s).toFloat()
        }
    }

    private fun baby(x: FloatArray, r: Random, level: Float) {
        var ph = 0.0
        val f0 = 380f + r.nextFloat() * 160f
        val burst = 0.55f + r.nextFloat() * 0.35f; val gap = 0.12f + r.nextFloat() * 0.15f
        val start = r.nextFloat() * 0.3f
        for (i in x.indices) {
            val t = i.toFloat() / SAMPLE_RATE
            val u = ((t + start) % (burst + gap))
            if (u > burst) continue
            val env = sin(PI * u / burst).toFloat()
            val f = f0 * (1f + 0.25f * sin(PI * u / burst).toFloat())         // rise then fall
            ph += 2 * PI * f / SAMPLE_RATE
            var s = 0.0
            for (h in 1..8) {
                val hz = f * h
                val formant = exp(-((hz - 1100f) * (hz - 1100f)) / (2 * 450f * 450f)) + 0.6 * exp(-((hz - 3000f) * (hz - 3000f)) / (2 * 600f * 600f))
                s += formant * sin(h * ph)
            }
            x[i] += (level * env * 0.9 * s).toFloat()
        }
    }
}
