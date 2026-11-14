package com.soundwitness.app

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import com.soundwitness.app.ui.C
import com.soundwitness.app.ui.DrivingScreen
import com.soundwitness.app.ui.ParkedScreen
import com.soundwitness.app.ui.text

class MainActivity : Activity() {
    private lateinit var c: Controller
    private lateinit var root: FrameLayout
    private var driving: DrivingScreen? = null
    private var parked: ParkedScreen? = null
    private var started = false
    private var lastDrivingMode: Boolean? = null
    private var lastRecordCount = -1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        root = FrameLayout(this).apply { setBackgroundColor(C.page) }
        setContentView(root)
        c = Controller(this)
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) begin()
        else {
            root.addView(text("마이크 권한이 필요합니다", 40f, C.ink2))
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) begin()
    }

    private fun begin() {
        if (started) return
        started = true
        c.onUi { render() }
        c.start()
        render()
        // adb shell am start -n com.soundwitness.app/.MainActivity --ez demo true
        if (intent?.getBooleanExtra("demo", false) == true) root.postDelayed({ c.toggleDemo() }, 4000)
    }

    /** driving: capture screen only. parked: reports. switches on gear / speed. */
    private fun render() {
        val drivingNow = c.isDriving()
        if (drivingNow != lastDrivingMode) {
            lastDrivingMode = drivingNow
            root.removeAllViews(); parked?.release()
            driving = null; parked = null
            val v: View = if (drivingNow) DrivingScreen(this, c).also { driving = it } else ParkedScreen(this, c).also { parked = it }
            root.addView(v, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        }
        driving?.update()
        val n = (c.engine.log.joinToString { it.label + it.detail } + c.detector.lastTest?.timeMs + c.learner.alerts().size).hashCode()
        if (parked != null && n != lastRecordCount) { lastRecordCount = n; parked?.refresh() }
    }

    override fun onDestroy() {
        if (started) c.stop()
        parked?.release()
        super.onDestroy()
    }
}
