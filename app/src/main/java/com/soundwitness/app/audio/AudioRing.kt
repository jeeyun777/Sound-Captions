package com.soundwitness.app.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.content.Context
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import kotlin.concurrent.thread

/**
 * Keeps the last [seconds] of microphone audio in memory only. Nothing reaches
 * storage unless [extract] is called for a user-requested clip.
 */
class AudioRing(private val context: Context, seconds: Int = 32) {
    private val tag = "AudioRing"
    private val buf = ShortArray(SAMPLE_RATE * seconds)
    @Volatile private var writePos = 0L   // total samples written
    @Volatile private var running = false
    private var record: AudioRecord? = null
    var sourceName = "-"; private set
    private val segmentListeners = mutableListOf<(ShortArray) -> Unit>()

    private val halfListeners = mutableListOf<(ShortArray) -> Unit>()
    private val last = ShortArray(SAMPLE_RATE); private var lastPos = 0; private var sinceHalf = 0
    /** while true, microphone data is ignored and [inject] supplies the audio (demo mode) */
    @Volatile var micMuted = false

    /** called with every new 1-second block, on the recording thread */
    fun onSecond(l: (ShortArray) -> Unit) { segmentListeners += l }

    /** called every 0.5 s with the latest 1 s of audio (event recognition) */
    fun onHalfSecond(l: (ShortArray) -> Unit) { halfListeners += l }

    /** demo audio enters here exactly like microphone audio */
    fun inject(chunk: ShortArray, n: Int = chunk.size) = push(chunk, n)

    private val second = ShortArray(SAMPLE_RATE); private var secPos = 0
    private var filled = 0      // samples in `last` since start/reset; no window until a full second

    /** start windows afresh (e.g. when switching between microphone and demo audio) */
    @Synchronized
    fun resetWindow() { filled = 0; sinceHalf = 0; secPos = 0 }

    @Synchronized
    private fun push(chunk: ShortArray, n: Int) {
        synchronized(buf) {
            for (i in 0 until n) buf[((writePos + i) % buf.size).toInt()] = chunk[i]
            writePos += n
        }
        var i = 0
        while (i < n) {
            val take = minOf(n - i, SAMPLE_RATE - secPos)
            System.arraycopy(chunk, i, second, secPos, take)
            secPos += take; i += take
            if (secPos == SAMPLE_RATE) {
                val copy = second.copyOf()
                segmentListeners.forEach { runCatching { it(copy) } }
                secPos = 0
            }
        }
        for (k in 0 until n) {
            last[lastPos] = chunk[k]; lastPos = (lastPos + 1) % SAMPLE_RATE
            if (filled < SAMPLE_RATE) { filled++; continue }
            if (++sinceHalf >= SAMPLE_RATE / 2) {
                sinceHalf = 0
                val win = ShortArray(SAMPLE_RATE) { last[(lastPos + it) % SAMPLE_RATE] }
                halfListeners.forEach { runCatching { it(win) } }
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val unprocessed = am.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        // prefer the least processed path so mechanical sounds are not filtered as noise
        val sources = buildList {
            if (unprocessed) add(MediaRecorder.AudioSource.UNPROCESSED to "UNPROCESSED")
            add(MediaRecorder.AudioSource.VOICE_RECOGNITION to "VOICE_RECOGNITION")
            add(MediaRecorder.AudioSource.MIC to "MIC")
        }
        for ((src, name) in sources) {
            val r = runCatching {
                AudioRecord(src, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf * 4, SAMPLE_RATE))
            }.getOrNull()
            if (r != null && r.state == AudioRecord.STATE_INITIALIZED) { record = r; sourceName = name; break }
            r?.release()
        }
        val rec = record ?: run { Log.e(tag, "no audio source"); return }
        running = true
        rec.startRecording()
        thread(name = "audio-ring", isDaemon = true) {
            val chunk = ShortArray(SAMPLE_RATE / 10)
            while (running) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n < 0) { Thread.sleep(50); continue }   // device error: back off instead of spinning
                if (n == 0 || micMuted) continue
                push(chunk, n)
            }
        }
    }

    fun stop() {
        running = false
        runCatching { record?.stop() }; record?.release(); record = null
    }

    val totalSamples: Long get() = writePos

    /** copy samples [fromSample, fromSample+len) by absolute sample index */
    fun extract(fromSample: Long, len: Int): ShortArray = synchronized(buf) {
        val out = ShortArray(len)
        val oldest = maxOf(0L, writePos - buf.size)
        for (i in 0 until len) {
            val idx = fromSample + i
            out[i] = if (idx in oldest until writePos) buf[(idx % buf.size).toInt()] else 0
        }
        out
    }

    companion object {
        fun writeWav(file: File, pcm: ShortArray) {
            file.parentFile?.mkdirs()
            RandomAccessFile(file, "rw").use { f ->
                f.setLength(0)
                val dataLen = pcm.size * 2
                fun i32(v: Int) { f.write(byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte())) }
                fun i16(v: Int) { f.write(byteArrayOf(v.toByte(), (v shr 8).toByte())) }
                f.writeBytes("RIFF"); i32(36 + dataLen); f.writeBytes("WAVE")
                f.writeBytes("fmt "); i32(16); i16(1); i16(1); i32(SAMPLE_RATE); i32(SAMPLE_RATE * 2); i16(2); i16(16)
                f.writeBytes("data"); i32(dataLen)
                val bytes = ByteArray(dataLen)
                for (i in pcm.indices) { bytes[2 * i] = pcm[i].toByte(); bytes[2 * i + 1] = (pcm[i].toInt() shr 8).toByte() }
                f.write(bytes)
            }
        }

        fun readWav(file: File): ShortArray {
            val b = file.readBytes()
            val n = (b.size - 44) / 2
            return ShortArray(n) { ((b[44 + 2 * it].toInt() and 0xff) or (b[45 + 2 * it].toInt() shl 8)).toShort() }
        }
    }
}
