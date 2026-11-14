package com.soundwitness.app.demo

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.events.EventSounds
import com.soundwitness.app.events.EventType
import com.soundwitness.app.learn.SyntheticSounds
import com.soundwitness.app.vehicle.VehicleSnapshot
import kotlin.concurrent.thread
import kotlin.random.Random

/**
 * Demo mode for the emulator and the demo video. Plays a scripted 80-second drive:
 * synthetic sounds are injected into the same pipeline as the microphone, and the
 * vehicle state follows the script so the interpretation and self-labeling can be
 * seen without a real car. The screen shows "시연 모드" while it runs.
 */
class DemoScenario(private val inject: (ShortArray) -> Unit, private val playAudio: Boolean = true) {

    companion object { const val LENGTH_S = 80f }

    @Volatile var running = false; private set
    @Volatile var t = 0f; private set          // seconds since start
    private var track: AudioTrack? = null

    fun start(onEnd: () -> Unit) {
        if (running) return
        running = true; t = 0f
        if (playAudio) track = runCatching {
            AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
                .setAudioFormat(AudioFormat.Builder().setSampleRate(SAMPLE_RATE).setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(SAMPLE_RATE).setTransferMode(AudioTrack.MODE_STREAM).build().also { it.play() }
        }.getOrNull()
        thread(name = "demo", isDaemon = true) {
            val r = Random(42)
            val chunk = SAMPLE_RATE / 10
            var n = 0L
            val t0 = System.nanoTime()
            while (running && t < LENGTH_S) {
                val pcm = render(n, chunk, r)
                inject(pcm)
                track?.write(pcm, 0, pcm.size)
                n += chunk; t = n.toFloat() / SAMPLE_RATE
                val due = t0 + (t * 1e9).toLong()
                val wait = (due - System.nanoTime()) / 1_000_000
                if (wait > 0) Thread.sleep(wait)
            }
            running = false
            runCatching { track?.stop(); track?.release() }; track = null
            onEnd()
        }
    }

    fun stop() { running = false }

    // ---------- sounds ----------

    /** demo audio for samples [n0, n0+len): cabin noise for the current speed plus the scripted events */
    fun render(n0: Long, len: Int, r: Random): ShortArray {
        val x = FloatArray(len)
        SyntheticSounds.cabin(x, r, if (speed(n0.toFloat() / SAMPLE_RATE) < 1f) 0.04f else 0.08f)
        addEvents(x, n0)
        return EventSounds.toPcm(x)
    }

    private val sirenRng = Random(7)
    private fun addEvents(x: FloatArray, n0: Long) {
        val sec0 = n0.toFloat() / SAMPLE_RATE
        // approaching then leaving siren, 4–14 s
        if (sec0 in 4f..14f) {
            EventSounds.siren(x, sirenRng, 1f, pattern = 0, phase = sec0 / 4f,
                gain = { i -> val s = sec0 + i.toFloat() / SAMPLE_RATE; if (s < 9f) 0.06f + 0.08f * (s - 4f) else (0.46f - 0.08f * (s - 9f)).coerceAtLeast(0.03f) },
                doppler = { i -> if (sec0 + i.toFloat() / SAMPLE_RATE < 9f) 1.03f else 0.97f })
        }
        // horns: at a red light (20 s) and during a lane change (29 s)
        if (sec0 in 20f..20.9f || sec0 in 29f..29.9f) mixClean(x, EventType.HORN, if (sec0 < 25f) 11 else 12, sec0 - if (sec0 < 25f) 20f else 29f, 0.25f)
        // child crying in the back, 36–41 s
        if (sec0 in 36f..41f) mixClean(x, EventType.BABY, 3 + (sec0 - 36f).toInt(), (sec0 - 36f) % 1f, 0.2f)
        // a siren inside a song, short and flat, 45–46.5 s (media playing)
        if (sec0 in 45f..46.5f) EventSounds.siren(x, sirenRng, 0.18f, pattern = 2, phase = 0f)
    }

    private fun mixClean(x: FloatArray, type: EventType, seed: Int, offsetS: Float, level: Float) {
        val e = EventSounds.clean(type, seed, 1f, level)
        val off = (offsetS * SAMPLE_RATE).toInt()
        for (i in x.indices) { val k = off + i; if (k in e.indices) x[i] += e[k] }
    }

    // ---------- scripted vehicle state ----------

    fun mediaPlaying(): Boolean = mediaAt(t)
    fun mediaAt(s: Float): Boolean = s in 44f..48f

    private fun speed(s: Float): Float = when {
        s < 7f -> 50f
        s < 10f -> 50f - 10f * (s - 7f)            // slowing for the siren
        s < 14f -> 20f
        s < 18f -> 20f - 5f * (s - 14f)
        s < 22f -> 0f                              // red light; horn at 20 s
        s < 26f -> 8f * (s - 22f)                  // moves off
        s < 50f -> 40f
        s < 76f -> 60f
        else -> 0f
    }

    fun snapshot(): VehicleSnapshot = snapshotAt(t)

    fun snapshotAt(s: Float, now: Long = System.currentTimeMillis()): VehicleSnapshot {
        val brake = s in 7f..10f || s in 17f..22f || s >= 76f
        val steer = when { s in 7.5f..10f -> 12f; s in 28f..30.5f -> 12f; else -> 0f }
        val signal = when { s in 27f..31f -> "RIGHT"; s in 50f..76f -> "LEFT"; else -> "OFF" }
        val gear = if (s >= 76f) "GEAR_PARK" else "GEAR_DRIVE"
        return VehicleSnapshot(now, brake, steer, speed(s), false, 0.3f, false, false, false, gear, signal, s in 34f..44f)
    }
}
