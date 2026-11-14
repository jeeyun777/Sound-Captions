package com.soundwitness.app.ui

import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import com.soundwitness.app.Controller
import com.soundwitness.app.events.EventType
import com.soundwitness.app.events.LogEntry
import com.soundwitness.app.learn.Learner
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Parked screen: what was shown today and whether each one was learned (decided
 * by the car, never asked), the self-test status, and the car's own sound changes.
 */
class ParkedScreen(private val ctx: Context, private val c: Controller) : LinearLayout(ctx) {
    private val body = FrameLayout(ctx)
    private val hm = SimpleDateFormat("HH:mm", Locale.KOREA)
    private var page = "log"          // log | mycar

    init {
        orientation = VERTICAL
        setBackgroundColor(C.page)
        addView(header(), lp(LayoutParams.MATCH_PARENT, ctx.px(116)))
        addView(View(ctx).apply { setBackgroundColor(C.line) }, lp(LayoutParams.MATCH_PARENT, 1))
        addView(body, lp(LayoutParams.MATCH_PARENT, 0, 1f))
        refresh()
    }

    private fun header(): View {
        val bar = ctx.hbox().apply { setPadding(ctx.px(64), 0, ctx.px(64), 0); setBackgroundColor(C.surface) }
        val logo = FrameLayout(ctx).apply {
            background = box(C.surface2, C.line)
            addView(WaveIcon(ctx, Color.WHITE), FrameLayout.LayoutParams(ctx.px(34), ctx.px(34), Gravity.CENTER))
        }
        bar.addView(logo, lp(ctx.px(56), ctx.px(56)).margins(r = ctx.px(22)))
        bar.addView(ctx.text("소리자막", 42f, C.ink, true).apply { setOnLongClickListener { c.toggleDemo(); true } })
        fun tab(label: String, key: String) = bar.addView(ctx.text(label, 38f, if (page == key) C.ink else C.ink3, page == key).apply {
            setPadding(ctx.px(30), ctx.px(10), ctx.px(30), ctx.px(10))
            if (page == key) background = box(C.surface2, C.line, 4f)
            setOnClickListener { page = key; rebuildHeader(); refresh() }
        }, lp(LayoutParams.WRAP_CONTENT).margins(l = ctx.px(24)))
        tab("오늘 들린 소리", "log"); tab("내 차 소리", "mycar")
        bar.addView(View(ctx), lp(0, 1, 1f))
        bar.addView(View(ctx).apply { background = box(C.ok, null, 999f) }, lp(ctx.px(14), ctx.px(14)).margins(r = ctx.px(12)))
        bar.addView(ctx.text("정차 중", 32f, C.ink2))
        bar.addView(ctx.text("P", 32f, C.ink, true).apply {
            setPadding(ctx.px(18), ctx.px(4), ctx.px(18), ctx.px(4)); background = box(Color.TRANSPARENT, C.ink, 4f, ctx.px(2))
        }, lp(LayoutParams.WRAP_CONTENT).margins(l = ctx.px(28)))
        return bar
    }

    private fun rebuildHeader() { removeViewAt(0); addView(header(), 0, lp(LayoutParams.MATCH_PARENT, ctx.px(116))) }

    fun refresh() {
        body.removeAllViews()
        body.addView(if (page == "log") logPage() else myCarPage(), FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    // ---------- today's sounds ----------

    private fun logPage(): View {
        val root = LinearLayout(ctx).apply { orientation = HORIZONTAL }
        val list = ctx.vbox().apply { setPadding(ctx.px(72), ctx.px(56), ctx.px(72), ctx.px(56)) }
        val head = ctx.hbox()
        head.addView(ctx.text("이번 주행에서 표시한 소리", 48f, C.ink, true))
        head.addView(ctx.text("  따로 확인하지 않아도 차가 스스로 학습합니다", 34f, C.ink3))
        list.addView(head, lp().margins(b = ctx.px(30)))
        val entries = c.engine.log.toList().reversed()
        if (entries.isEmpty()) list.addView(ctx.text("아직 표시한 소리가 없습니다. 제목을 길게 누르면 시연이 시작됩니다.", 34f, C.ink3))
        entries.take(12).forEach { list.addView(row(it), lp().margins(b = ctx.px(22))) }
        root.addView(ScrollView(ctx).apply { addView(list) }, lp(0, LayoutParams.MATCH_PARENT, 1f))

        val side = ctx.vbox().apply { setBackgroundColor(C.surface); setPadding(ctx.px(56), ctx.px(60), ctx.px(56), ctx.px(50)) }
        fun stat(k: String, v: String, big: Float = 76f) { side.addView(ctx.text(k, 34f, C.ink2)); side.addView(ctx.text(v, big, C.ink, true), lp().margins(t = ctx.px(6), b = ctx.px(40))) }
        stat("이 차 실내 소음 학습", "%,d구간".format(c.detector.bgAdapted))
        stat("오늘 학습 반영", "${entries.count { it.label == "학습 반영" }}건")
        val t = c.detector.lastTest
        side.addView(ctx.text("자가 시험", 34f, C.ink2))
        if (t == null) side.addView(ctx.text("준비 중", 44f, C.ink, true))
        else {
            val name = mapOf(EventType.SIREN to "사이렌", EventType.HORN to "경적", EventType.BABY to "아이 울음")
            side.addView(ctx.text(t.recall.entries.joinToString("  ") { "${name[it.key]} ${it.value}/${t.perType}" }, 38f, C.ink, true), lp().margins(t = ctx.px(8)))
            side.addView(ctx.text("평소 소음 오인 ${t.falseAlarms}/${t.bgWindows}", 34f, C.ink2), lp().margins(t = ctx.px(10)))
            if (t.rolledBack) side.addView(ctx.text("성능이 떨어져 직전 학습을 되돌렸습니다", 32f, C.warnInk, true), lp().margins(t = ctx.px(12)))
        }
        side.addView(View(ctx), lp(1, 0, 1f))
        side.addView(ctx.text("운전자에게 묻지 않고\n운전자 반응과 소리 흐름으로 학습\n원음은 저장하지 않음 · 차 안에서 처리", 30f, C.ink3).apply { setLineSpacing(0f, 1.5f) })
        root.addView(side, lp(ctx.px(640), LayoutParams.MATCH_PARENT))
        return root
    }

    private fun row(e: LogEntry): View {
        val r = LinearLayout(ctx).apply {
            orientation = HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            background = box(C.surface, C.line); setPadding(ctx.px(40), ctx.px(28), ctx.px(40), ctx.px(28))
        }
        r.addView(ctx.text(hm.format(Date(e.timeMs)), 40f, C.ink2), lp(ctx.px(150)))
        val mid = ctx.vbox()
        mid.addView(ctx.text(e.title, 44f, if (e.kind == "SIREN") C.warnInk else C.ink, true))
        mid.addView(ctx.text(e.detail, 32f, C.ink3), lp().margins(t = ctx.px(10)))
        r.addView(mid, lp(0, LayoutParams.WRAP_CONTENT, 1f))
        val lab = ctx.vbox().apply { gravity = Gravity.END }
        val ok = e.label == "학습 반영"
        lab.addView(ctx.text(e.label, 32f, if (ok) C.ok else C.ink2, true).apply {
            setPadding(ctx.px(24), ctx.px(10), ctx.px(24), ctx.px(10)); background = box(Color.TRANSPARENT, if (ok) C.ok else C.line, 4f)
        }, lp(LayoutParams.WRAP_CONTENT))
        if (e.reason.isNotEmpty()) lab.addView(ctx.text(e.reason, 28f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(10)))
        r.addView(lab, lp(ctx.px(560)))
        return r
    }

    // ---------- my car's sound (Learner) ----------

    private fun myCarPage(): View {
        val a = c.learner.alerts().firstOrNull() ?: return ctx.vbox().apply {
            gravity = Gravity.CENTER
            addView(ctx.text("내 차 소리는 평소 범위 안에 있습니다", 52f, C.ink2, true))
            addView(ctx.text("정상 주행 소리 학습 ${c.learner.learnedSegments}구간 · 대화로 판단해 제외 ${c.learner.skippedSpeech}구간", 32f, C.ink3), lp(LayoutParams.WRAP_CONTENT).margins(t = ctx.px(24)))
        }
        return alertDetail(a)
    }

    private fun alertDetail(a: Learner.Alert): View {
        val root = LinearLayout(ctx).apply { orientation = HORIZONTAL; setPadding(ctx.px(72), ctx.px(48), ctx.px(72), ctx.px(48)) }
        val card = ctx.vbox().apply { background = box(C.surface, C.line); setPadding(ctx.px(48), ctx.px(40), ctx.px(48), ctx.px(40)) }
        card.addView(ctx.hbox().apply {
            background = box(C.warnBg, C.warnLine, 4f); setPadding(ctx.px(22), ctx.px(10), ctx.px(22), ctx.px(10))
            addView(WarnTriangle(ctx), lp(ctx.px(34), ctx.px(34)).margins(r = ctx.px(14)))
            addView(ctx.text("점검 권장", 32f, C.warnInk, true))
        }, lp(LayoutParams.WRAP_CONTENT))
        val khz = { hz: Int -> if (hz >= 1000) "%.1f".format(hz / 1000f).removeSuffix(".0") + "k" else "$hz" }
        card.addView(ctx.text("${a.state.label} 시 ${khz(a.bandLowHz)}–${khz(a.bandHighHz)}Hz 성분이 커졌습니다", 54f, C.ink, true), lp().margins(t = ctx.px(22)))
        card.addView(ctx.text("최근 ${a.window}구간 중 ${a.outOfRange}구간에서 이 차량의 평소 범위를 벗어났습니다. 고음 경고음은 나이가 들수록 잘 들리지 않아 화면으로 알려 드립니다.", 36f, C.ink2).apply { setLineSpacing(0f, 1.4f) }, lp().margins(t = ctx.px(22)))
        card.addView(ctx.text("${a.state.label} 시 ${khz(a.bandLowHz)}–${khz(a.bandHighHz)}Hz 성분, 평소 대비 변화(dB)", 30f, C.ink2, true), lp().margins(t = ctx.px(22)))
        card.addView(TrendView(ctx).apply { values = c.learner.history[a.state]?.toList() ?: emptyList() }, lp(LayoutParams.MATCH_PARENT, ctx.px(520)))
        root.addView(card, lp(0, LayoutParams.MATCH_PARENT, 1f).margins(r = ctx.px(48)))

        val side = ctx.vbox().apply { background = box(C.surface, C.line) }
        fun fact(k: String, v: String, small: String) = side.addView(ctx.vbox().apply {
            setPadding(ctx.px(40), ctx.px(32), ctx.px(40), ctx.px(32))
            addView(ctx.text(k, 30f, C.ink2))
            addView(ctx.hbox().apply { addView(ctx.text(v, 52f, C.ink, true)); addView(ctx.text(" $small", 30f, C.ink2)) }, lp().margins(t = ctx.px(6)))
        })
        fact("평소 범위 이탈", "${a.outOfRange}", "/ 최근 ${a.window}구간")
        fact("변화량", "+%.1f".format(a.changeDb), "dB")
        c.learner.levelShift(a.state)?.let { fact("전체 소음 변화 (판단에서 제외)", "%+.1f".format(it), "dB") }
        side.addView(View(ctx), lp(1, 0, 1f))
        side.addView(ctx.vbox().apply {
            setPadding(ctx.px(40), ctx.px(32), ctx.px(40), ctx.px(32))
            addView(ctx.button("정비소 안내", primary = true) { c.navigateToRepairShop(ctx) }, lp().margins(b = ctx.px(20)))
            addView(ctx.button("정비 완료 표시") { confirmRepairDone() }, lp())
        })
        root.addView(side, lp(ctx.px(720), LayoutParams.MATCH_PARENT))
        return root
    }

    private fun confirmRepairDone() {
        AlertDialog.Builder(ctx).setTitle("정비 완료 표시")
            .setMessage("지금부터 이 차량의 평소 소리를 다시 학습합니다.")
            .setPositiveButton("확인") { _, _ -> c.learner.resetAfterRepair(); refresh() }
            .setNegativeButton("취소", null).show()
    }

    fun release() {}
}
