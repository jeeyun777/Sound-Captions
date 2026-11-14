package com.soundwitness.app.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

const val SAMPLE_RATE = 16_000
const val FFT_N = 512
const val HOP = 256

object Dsp {
    private val window = FloatArray(FFT_N) { (0.5 - 0.5 * cos(2 * PI * it / (FFT_N - 1))).toFloat() }
    private val cosT = FloatArray(FFT_N / 2) { cos(2 * PI * it / FFT_N).toFloat() }
    private val sinT = FloatArray(FFT_N / 2) { -sin(2 * PI * it / FFT_N).toFloat() }

    /** power spectrum (FFT_N/2+1 bins) of FFT_N samples starting at [off] */
    fun power(x: ShortArray, off: Int, out: FloatArray, re: FloatArray, im: FloatArray) {
        for (i in 0 until FFT_N) {
            val v = if (off + i < x.size) x[off + i] / 32768f else 0f
            re[i] = v * window[i]; im[i] = 0f
        }
        fft(re, im)
        for (k in 0..FFT_N / 2) out[k] = re[k] * re[k] + im[k] * im[k]
    }

    private fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t }
        }
        var len = 2
        while (len <= n) {
            val step = n / len
            for (i in 0 until n step len) {
                for (k in 0 until len / 2) {
                    val c = cosT[k * step]; val s = sinT[k * step]
                    val a = i + k; val b = a + len / 2
                    val tr = re[b] * c - im[b] * s
                    val ti = re[b] * s + im[b] * c
                    re[b] = re[a] - tr; im[b] = im[a] - ti
                    re[a] += tr; im[a] += ti
                }
            }
            len = len shl 1
        }
    }

    /** log-spaced band edges (in FFT bins) between [lowHz] and [highHz] */
    fun bandEdges(bands: Int, lowHz: Float = 150f, highHz: Float = 7800f): IntArray {
        val binHz = SAMPLE_RATE.toFloat() / FFT_N
        return IntArray(bands + 1) {
            val hz = lowHz * exp(ln(highHz / lowHz) * it / bands)
            (hz / binHz).toInt().coerceIn(1, FFT_N / 2)
        }
    }

    fun bandCenterHz(edges: IntArray, b: Int): Float {
        val binHz = SAMPLE_RATE.toFloat() / FFT_N
        return sqrt(edges[b].toFloat() * max(edges[b + 1], edges[b] + 1)) * binHz
    }

    /**
     * Band energies in dB, [bands] x frames. Each frame is one HOP step.
     */
    fun bandFrames(x: ShortArray, start: Int, length: Int, edges: IntArray): Array<FloatArray> {
        val bands = edges.size - 1
        val frames = max(1, (length - FFT_N) / HOP + 1)
        val re = FloatArray(FFT_N); val im = FloatArray(FFT_N); val p = FloatArray(FFT_N / 2 + 1)
        val out = Array(bands) { FloatArray(frames) }
        for (f in 0 until frames) {
            power(x, start + f * HOP, p, re, im)
            for (b in 0 until bands) {
                var e = 0f
                val hi = max(edges[b + 1], edges[b] + 1)
                for (k in edges[b] until hi) e += p[k]
                out[b][f] = (10 * log10(e / (hi - edges[b]) + 1e-10)).toFloat()
            }
        }
        return out
    }

    /** average the frame axis down to [cols] columns */
    fun poolTime(m: Array<FloatArray>, cols: Int): Array<FloatArray> = Array(m.size) { b ->
        val src = m[b]
        FloatArray(cols) { c ->
            val a = c * src.size / cols; val z = max(a + 1, (c + 1) * src.size / cols)
            var s = 0f; for (i in a until min(z, src.size)) s += src[i]; s / (min(z, src.size) - a).coerceAtLeast(1)
        }
    }

    fun rmsDb(x: ShortArray, start: Int, length: Int): Float {
        var s = 0.0
        val end = min(x.size, start + length)
        for (i in start until end) { val v = x[i] / 32768.0; s += v * v }
        return (10 * log10(s / max(1, end - start) + 1e-12)).toFloat()
    }

    /**
     * Rough speech check: speech puts most energy in 300–3400 Hz and its band
     * energy changes quickly (syllable rate). Segments that look like speech are
     * not used for learning.
     */
    fun looksLikeSpeech(bandDb: Array<FloatArray>, edges: IntArray): Boolean {
        val bands = bandDb.size
        var voice = 0.0; var total = 0.0
        val frames = bandDb[0].size
        val energyPerFrame = FloatArray(frames)
        for (b in 0 until bands) {
            val hz = bandCenterHz(edges, b)
            for (f in 0 until frames) {
                val e = Math.pow(10.0, bandDb[b][f] / 10.0)
                total += e
                if (hz in 300f..3400f) { voice += e; energyPerFrame[f] += e.toFloat() }
            }
        }
        if (total <= 0) return false
        val ratio = voice / total
        // modulation: how much voice-band energy swings frame to frame
        var swing = 0.0; var mean = 0.0
        for (f in 0 until frames) mean += energyPerFrame[f]
        mean /= frames
        for (f in 1 until frames) swing += kotlin.math.abs(energyPerFrame[f] - energyPerFrame[f - 1])
        val mod = if (mean > 0) swing / (frames * mean) else 0.0
        return ratio > 0.65 && mod > 0.35
    }
}
