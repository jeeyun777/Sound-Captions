package com.soundwitness.app.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import com.soundwitness.app.Controller
import com.soundwitness.app.events.Alert
import com.soundwitness.app.events.Interpreter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Driving screen: nothing to touch. Idle shows that the app is listening; an
 * alert shows one pictogram, one title and one line, then disappears by itself.
 */
class DrivingScreen(private val ctx: Context, private val c: Controller) : LinearLayout(ctx) {
    private val stateText = ctx.text("", 32f, C.ink2)
    private val gearText = ctx.text("D", 32f, C.ink, true)
    private val demoText = ctx.text("", 30f, C.warnInk, true)
    private val center = FrameLayout(ctx)
    private var shownKey = ""
    private val hm = SimpleDateFormat("HH:mm", Locale.KOREA)

    init {
        orientation = VERTICAL
        setBackgroundColor(C.page)
        addView(header(), lp(LayoutParams.MATCH_PARENT, ctx.px(116)))
        addView(View(ctx).apply { setBackgroundColor(C.line) }, lp(LayoutParams.MATCH_PARENT, 1))
        addView(center, lp(LayoutParams.MATCH_PARENT, 0, 1f))
        update()
    }

    private fun header(): LinearLayout {
        val bar = ctx.hbox().apply { setPadding(ctx.px(64), 0, ctx.px(64), 0); setBackgroundColor(C.surface) }
        val logo = FrameLayout(ctx).apply {
            background = box(C.surface2, C.line)
            addView(WaveIcon(ctx, Color.WHITE), FrameLayout.LayoutParams(ctx.px(34), ctx.px(34), Gravity.CENTER))
        }
        bar.addView(logo, lp(ctx.px(56), ctx.px(56)).margins(r = ctx.px(22)))
        bar.addView(ctx.text("소리자막", 42f, C.ink, true).apply { setOnLongClickListener { c.toggleDemo(); true } })
        bar.addView(demoText, lp(LayoutParams.WRAP_CONTENT).margins(l = ctx.px(28)))
        bar.addView(View(ctx), lp(0, 1, 1f))
        bar.addView(View(ctx).apply { background = box(C.ok, null, 999f) }, lp(ctx.px(14), ctx.px(14)).margins(r = ctx.px(12)))
        bar.addView(stateText)
        gearText.setPadding(ctx.px(18), ctx.px(4), ctx.px(18), ctx.px(4))
        gearText.background = box(Color.TRANSPARENT, C.ink, 4f, ctx.px(2))
        bar.addView(gearText, lp(LayoutParams.WRAP_CONTENT).margins(l = ctx.px(28)))
        return bar
    }

    fun update() {
        val snap = c.snapshot()
        stateText.text = if (c.ready) "듣는 중 · 차량 내 처리" else "준비 중"
        gearText.text = Interpreter.gearLabel(snap.gear)
        demoText.text = when { c.demoRunning -> "시연 모드 ${c.demoTime.toInt()}초"; c.demoPending -> "시연 준비 중"; else -> "" }
        val a = c.currentAlert()
        val key = a?.let { "${it.kind}|${it.title}|${it.trend}|${it.sub}" } ?: "idle|${c.engine.log.lastOrNull()?.timeMs}"
        if (key == shownKey) return
        shownKey = key
        center.removeAllViews()
        center.addView(if (a != null) alertView(a) else idleView(), FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    private fun idleView(): View {
        val col = ctx.vbox().apply { gravity = Gravity.CENTER }
        col.addView(ctx.text("도로의 소리를 듣고 있습니다", 56f, C.ink2, true))
        col.addView(ctx.text("필요한 소리가 들리면 이곳에 한 줄로 알려 드립니다", 34f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(28)))
        c.engine.log.lastOrNull()?.let { col.addView(ctx.text("최근 · ${hm.format(Date(it.timeMs))} ${it.title}", 32f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(60))) }
        return col
    }

    private fun alertView(a: Alert): View {
        val col = ctx.vbox().apply { gravity = Gravity.CENTER }
        val siren = a.kind == "SIREN"
        val accent = when (a.kind) { "SIREN" -> C.warnInk; "HORN" -> Color.parseColor("#9bb2d9"); else -> C.ink }
        val card = LinearLayout(ctx).apply {
            orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(ctx.px(72), ctx.px(64), ctx.px(72), ctx.px(64))
            background = if (siren) box(Color.parseColor("#1e1a12"), Color.parseColor("#b8862c"), 8f, ctx.px(3))
                         else box(C.surface, if (a.kind == "HORN") C.accent else C.line, 8f, ctx.px(3))
        }
        val icoBox = FrameLayout(ctx).apply { background = box(if (siren) C.warnBg else if (a.kind == "HORN") C.accentSoft else C.surface2, null, 6f) }
        icoBox.addView(EventIcon(ctx, a.kind, accent), FrameLayout.LayoutParams(ctx.px(170), ctx.px(170), Gravity.CENTER))
        card.addView(icoBox, lp(ctx.px(260), ctx.px(260)).margins(r = ctx.px(56)))
        val texts = ctx.vbox()
        texts.addView(ctx.text(a.title, 100f, C.ink, true))
        texts.addView(ctx.text(a.sub, 54f, C.ink2, true), lp().margins(t = ctx.px(22)))
        card.addView(texts, lp(ctx.px(1100)))
        col.addView(card, lp(ctx.px(1500)))
        if (siren && a.trend != 0) {
            val row = ctx.hbox()
            val bars = ctx.hbox().apply { gravity = Gravity.BOTTOM }
            val hs = if (a.trend > 0) listOf(30, 42, 55, 70, 84, 100) else listOf(100, 84, 70, 55, 42, 30)
            hs.forEachIndexed { i, h ->
                val dim = if (a.trend > 0) i < 2 else i > 3
                bars.addView(View(ctx).apply { setBackgroundColor(if (dim) Color.parseColor("#4a3d22") else Color.parseColor("#b8862c")) }, lp(ctx.px(34), ctx.px(120) * h / 100).margins(r = ctx.px(14)))
            }
            row.addView(bars, lp(LayoutParams.WRAP_CONTENT, ctx.px(120)).margins(r = ctx.px(26)))
            row.addView(ctx.text(if (a.trend > 0) "소리가 커지는 중 (다가옴)" else "소리가 작아지는 중 (멀어짐)", 46f, C.warnInk, true))
            col.addView(row, lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(56)))
            col.addView(ctx.text("방향은 표시하지 않습니다", 32f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(36)))
        }
        if (a.chips.isNotEmpty()) {
            col.addView(ctx.text("이렇게 판단했습니다", 32f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(56)))
            val row = ctx.hbox()
            a.chips.forEach { (k, v) ->
                row.addView(ctx.text("$k $v", 38f, C.ink2, true).apply {
                    setPadding(ctx.px(30), ctx.px(16), ctx.px(30), ctx.px(16)); background = box(C.surface, C.line, 4f)
                }, lp(LayoutParams.WRAP_CONTENT).margins(r = ctx.px(20)))
            }
            col.addView(row, lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(20)))
        }
        return col
    }
}
