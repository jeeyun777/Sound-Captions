package com.soundwitness.app

import android.content.Context
import android.media.AudioManager
import android.os.Handler
import android.os.Looper
import com.soundwitness.app.audio.AudioRing
import com.soundwitness.app.audio.Dsp
import com.soundwitness.app.demo.DemoScenario
import com.soundwitness.app.events.Alert
import com.soundwitness.app.events.EventDetector
import com.soundwitness.app.events.EventEngine
import com.soundwitness.app.learn.Encoder
import com.soundwitness.app.learn.Learner
import com.soundwitness.app.vehicle.VehicleMonitor
import com.soundwitness.app.vehicle.VehicleSnapshot
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

/**
 * 소리자막 wiring: microphone (or demo audio) → event recognition every 0.5 s →
 * vehicle-context interpretation → one alert on the driving screen; plus the
 * car's own sound changes (Learner) and the self test that guards learning.
 */
class Controller(context: Context) {
    private val appContext = context.applicationContext

    val vehicle = VehicleMonitor(context)
    val audio = AudioRing(context)
    val learner = Learner(context)
    val detector = EventDetector()
    val engine = EventEngine(detector)
    private val audioManager = appContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val uiListeners = mutableListOf<() -> Unit>()
    private val eventsFile = File(appContext.filesDir, "events.json")
    private val logFile = File(appContext.filesDir, "events_log.json")

    @Volatile var ready = false; private set
    @Volatile var level = -90f; private set
    private var demo: DemoScenario? = null
    val demoRunning get() = demo?.running == true
    val demoTime get() = demo?.t ?: 0f
    @Volatile var demoPending = false; private set
    private var demoT0 = 0L
    private var wasDriving = false

    fun onUi(l: () -> Unit) { uiListeners += l }
    private fun notifyUi() = main.post { uiListeners.forEach { it() } }

    private val ticker = object : Runnable {
        override fun run() {
            vehicle.tick()
            val s = snapshot()
            val now = now()
            worker.execute { engine.tick(now, s) }
            val driving = isDriving(s)
            if (wasDriving && !driving) worker.execute { afterDrive() }
            wasDriving = driving
            notifyUi()
            main.postDelayed(this, 500)
        }
    }

    fun start() {
        learner.load()
        worker.execute {
            detector.trainPrototypes()
            runCatching { if (eventsFile.exists()) detector.load(JSONObject(eventsFile.readText())) }
            runCatching { if (logFile.exists()) engine.loadLog(JSONArray(logFile.readText())) }
            learner.trainCategories()
            detector.selfTest()
            ready = true; notifyUi()
            if (demoPending) main.post { demoPending = false; toggleDemo() }
        }
        engine.onLearned = { save() }
        vehicle.start()
        audio.onHalfSecond { pcm ->
            // vehicle state and time are taken when the window is cut, not when the worker gets to it
            val snap = snapshot(); val media = mediaPlaying(); val now = now()
            worker.execute { onWindow(pcm, snap, media, now) }
        }
        audio.onSecond { pcm ->
            val snap = snapshot()
            if (isDriving(snap)) worker.execute { learner.onSegment(pcm, snap) }
        }
        audio.start()
        main.post(ticker)
    }

    fun stop() {
        main.removeCallbacks(ticker)
        demo?.stop()
        audio.stop(); vehicle.stop(); learner.save(); save()
    }

    fun snapshot(): VehicleSnapshot = demo?.takeIf { it.running }?.snapshot() ?: vehicle.snapshot()

    /** clock for events: the demo timeline while the demo runs (so a slow device does not stretch it) */
    fun now(): Long = demo?.takeIf { it.running }?.let { demoT0 + (it.t * 1000).toLong() } ?: System.currentTimeMillis()

    fun mediaPlaying(): Boolean = demo?.takeIf { it.running }?.mediaPlaying() ?: audioManager.isMusicActive

    fun isDriving(s: VehicleSnapshot = snapshot()): Boolean = when (s.gear) {
        "GEAR_PARK" -> false
        null, "GEAR_UNKNOWN" -> (s.speedKmh ?: 0f) > 1f
        else -> true
    }

    /** rear-seat child feature: on when someone sits in the back, or when the car does not report it */
    private fun rearAllowed(s: VehicleSnapshot) = s.rearOccupied != false

    private fun onWindow(pcm: ShortArray, snap: VehicleSnapshot, media: Boolean, now: Long) {
        if (!ready) return
        level = Dsp.rmsDb(pcm, 0, pcm.size)
        val frames = Dsp.bandFrames(pcm, 0, pcm.size, Encoder.edges)
        if (Dsp.looksLikeSpeech(frames, Encoder.edges)) return      // speech is never processed
        val w = detector.classify(pcm, media)
        engine.push(w, now, snap, media, rearAllowed(snap))
        if (w.type == null && !media && isDriving(snap)) detector.learnBackground(pcm, w)
    }

    /** on parking: self test guards what was learned during the drive */
    private fun afterDrive() {
        if (!ready) return
        detector.selfTest()
        save(); notifyUi()
    }

    fun currentAlert(): Alert? = engine.alert

    fun save() {
        runCatching { eventsFile.writeText(detector.toJson().toString()) }
        runCatching { logFile.writeText(engine.logJson().toString()) }
    }

    // ---------- demo ----------

    fun toggleDemo() {
        if (demoRunning) { demo?.stop(); audio.micMuted = false; audio.resetWindow(); return }
        if (!ready) { demoPending = !demoPending; notifyUi(); return }      // start once models are ready
        engine.log.clear()
        demoT0 = System.currentTimeMillis()
        audio.micMuted = true
        audio.resetWindow()
        demo = DemoScenario({ audio.inject(it) }).also {
            it.start {
                audio.micMuted = false
                audio.resetWindow()
                worker.execute { afterDrive() }
            }
        }
        notifyUi()
    }

    /** start navigation to a nearby repair shop; replaced by NaviHelper route search when wired */
    fun navigateToRepairShop(ctx: Context) {
        val i = android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse("geo:0,0?q=" + android.net.Uri.encode("자동차 정비소")))
        runCatching { ctx.startActivity(i) }.onFailure {
            android.widget.Toast.makeText(ctx, "내비게이션 앱을 찾지 못했습니다", android.widget.Toast.LENGTH_SHORT).show()
        }
    }
}
