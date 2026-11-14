package com.soundwitness.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView

/** colors from the approved mockups (mockups/common.css) */
object C {
    val page = Color.parseColor("#0e0f10")
    val surface = Color.parseColor("#17181a")
    val surface2 = Color.parseColor("#202124")
    val line = Color.parseColor("#2f3135")
    val ink = Color.parseColor("#ebebe8")
    val ink2 = Color.parseColor("#b3b5b8")
    val ink3 = Color.parseColor("#8a8d92")
    val accent = Color.parseColor("#3f68b5")
    val accentSoft = Color.parseColor("#1d2638")
    val warnInk = Color.parseColor("#e3ad55")
    val warnBg = Color.parseColor("#2a2214")
    val warnLine = Color.parseColor("#8a6624")
    val ok = Color.parseColor("#5c9a6a")
    val bar = Color.parseColor("#5a80c4")
    val chart = Color.parseColor("#6f92cf")
}

fun Context.px(v: Int): Int = (v * resources.displayMetrics.density).toInt()

fun box(fill: Int, stroke: Int? = null, radius: Float = 6f, strokeW: Int = 1) = GradientDrawable().apply {
    setColor(fill); cornerRadius = radius
    if (stroke != null) setStroke(strokeW, stroke)
}

fun Context.text(s: String, size: Float, color: Int = C.ink, bold: Boolean = false): TextView = TextView(this).apply {
    text = s; textSize = size; setTextColor(color)
    typeface = Typeface.create(Typeface.DEFAULT, if (bold) Typeface.BOLD else Typeface.NORMAL)
    includeFontPadding = false
}

fun Context.vbox(gap: Int = 0) = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; if (gap > 0) dividerPadding = gap }
fun Context.hbox() = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }

fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT, weight: Float = 0f) =
    LinearLayout.LayoutParams(w, h, weight)

fun LinearLayout.LayoutParams.margins(l: Int = 0, t: Int = 0, r: Int = 0, b: Int = 0) = apply { setMargins(l, t, r, b) }

fun Context.button(label: String, primary: Boolean = false, onClick: () -> Unit): TextView = text(label, 38f, if (primary) Color.WHITE else C.ink, true).apply {
    gravity = Gravity.CENTER
    setPadding(px(52), 0, px(52), 0)
    background = if (primary) box(C.accent, C.accent) else box(C.surface, C.line)
    minHeight = px(104)
    setOnClickListener { onClick() }
}

/** top bar: brand on the left, state + gear on the right */
fun Context.topBar(sub: String?, state: String, gear: String, ok: Boolean): View {
    val bar = hbox().apply {
        setPadding(px(64), 0, px(64), 0)
        setBackgroundColor(C.surface)
    }
    val logo = FrameLayout(this).apply {
        background = box(C.surface2, C.line)
        addView(WaveIcon(context, Color.WHITE), FrameLayout.LayoutParams(px(34), px(34), Gravity.CENTER))
    }
    bar.addView(logo, lp(px(56), px(56)).margins(r = px(22)))
    bar.addView(text("소리증인", 42f, C.ink, true))
    if (sub != null) bar.addView(text("  $sub", 42f, C.ink2))
    bar.addView(View(this), lp(0, 1, 1f))
    if (ok) bar.addView(View(this).apply { background = box(C.ok, null, 999f) }, lp(px(14), px(14)).margins(r = px(12)))
    bar.addView(text(state, 32f, C.ink2))
    bar.addView(text(gear, 32f, C.ink, true).apply {
        setPadding(px(18), px(4), px(18), px(4)); background = box(Color.TRANSPARENT, C.ink, 4f, px(2))
    }, lp(ViewGroup.LayoutParams.WRAP_CONTENT).margins(l = px(28)))
    return bar
}

class WaveIcon(ctx: Context, private val color: Int) : View(ctx) {
    private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND }
    override fun onDraw(c: Canvas) {
        p.color = color; p.strokeWidth = width / 10f
        val u = width / 24f
        val lines = listOf(4f to 12f to 12f, 8f to 8f to 16f, 12f to 5f to 19f, 16f to 9f to 15f, 20f to 11f to 13f)
        for ((xy, y2) in lines) c.drawLine(xy.first * u, xy.second * u, xy.first * u, y2 * u, p)
    }
}

/** spectrogram bitmap from RecordStore.writeSpectrogram, with the press moment dashed */
class SpectrogramView(ctx: Context) : View(ctx) {
    private val ramp = listOf("#141516", "#1a2130", "#22304a", "#2c4166", "#3a5688", "#4c6ea8", "#6b8bc2", "#9bb2d9", "#d0dcef").map { Color.parseColor(it) }
    private var data: Triple<Int, Int, ByteArray>? = null
    private var pressFrac = 2f / 3f
    private val p = Paint()
    private val dash = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = C.ink; strokeWidth = 2f; style = Paint.Style.STROKE; pathEffect = DashPathEffect(floatArrayOf(10f, 8f), 0f) }

    fun set(d: Triple<Int, Int, ByteArray>?, press: Float) { data = d; pressFrac = press; invalidate() }

    override fun onDraw(c: Canvas) {
        c.drawColor(ramp[0])
        val (rows, cols, b) = data ?: return
        val cw = width.toFloat() / cols; val rh = height.toFloat() / rows
        for (r in 0 until rows) for (col in 0 until cols) {
            val v = b[r * cols + col].toInt() and 0xff
            p.color = ramp[(v * (ramp.size - 1) + 127) / 255]
            c.drawRect(col * cw, r * rh, (col + 1) * cw + 1, (r + 1) * rh + 1, p)
        }
        val x = width * pressFrac
        c.drawPath(Path().apply { moveTo(x, 0f); lineTo(x, height.toFloat()) }, dash)
    }
}

/** line chart of band change vs the normal band (alert screen) */
class TrendView(ctx: Context) : View(ctx) {
    var values: List<Float> = emptyList(); set(v) { field = v; invalidate() }
    private val grid = Paint().apply { color = Color.parseColor("#2a2c30"); strokeWidth = 2f }
    private val band = Paint().apply { color = Color.argb(15, 255, 255, 255) }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = C.chart; strokeWidth = 3f; style = Paint.Style.STROKE; strokeJoin = Paint.Join.ROUND }
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = C.chart }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = C.ink3; textSize = 26f }
    override fun onDraw(c: Canvas) {
        val l = 100f; val r = 30f; val t = 30f; val b = 70f
        val yMin = -2f; val yMax = 7f
        fun y(v: Float) = t + (height - t - b) * (1 - (v - yMin) / (yMax - yMin))
        for (v in listOf(-2, 0, 2, 4, 6)) {
            c.drawLine(l, y(v.toFloat()), width - r, y(v.toFloat()), grid)
            label.textAlign = Paint.Align.RIGHT; c.drawText(if (v > 0) "+$v" else "$v", l - 16, y(v.toFloat()) + 9, label)
        }
        c.drawRect(l, y(1.2f), width - r, y(-1.2f), band)
        label.textAlign = Paint.Align.LEFT; label.color = C.ink2; c.drawText("평소 범위", l + 16, y(1.2f) + 34, label); label.color = C.ink3
        if (values.size < 2) return
        val n = values.size
        fun x(i: Int) = l + (width - l - r) * i / (n - 1)
        val path = Path()
        values.forEachIndexed { i, v -> val yy = y(v.coerceIn(yMin, yMax)); if (i == 0) path.moveTo(x(i), yy) else path.lineTo(x(i), yy) }
        c.drawPath(path, line)
        values.forEachIndexed { i, v -> if (v > 1.2f) c.drawCircle(x(i), y(v.coerceIn(yMin, yMax)), 7f, dot) }
        label.textAlign = Paint.Align.LEFT; c.drawText("이전", l, height - 22f, label)
        label.textAlign = Paint.Align.RIGHT; c.drawText("최근", width - r, height - 22f, label)
    }
}

class WarnTriangle(ctx: Context) : View(ctx) {
    private val p = Paint(Paint.ANTI_ALIAS_FLAG)
    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        p.color = Color.parseColor("#b8862c")
        c.drawPath(Path().apply { moveTo(w / 2, h * 0.1f); lineTo(w * 0.06f, h * 0.88f); lineTo(w * 0.94f, h * 0.88f); close() }, p)
        p.color = C.page
        c.drawRect(w * 0.46f, h * 0.38f, w * 0.54f, h * 0.62f, p)
        c.drawRect(w * 0.46f, h * 0.69f, w * 0.54f, h * 0.77f, p)
    }
}

/** pictograms for the event alerts: SIREN, HORN, BABY, TURN */
class EventIcon(ctx: Context, private val kind: String, private val color: Int) : View(ctx) {
    private val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND }
    override fun onDraw(c: Canvas) {
        val u = width / 24f
        p.color = color; p.strokeWidth = 1.8f * u; p.style = Paint.Style.STROKE
        fun line(x1: Float, y1: Float, x2: Float, y2: Float) = c.drawLine(x1 * u, y1 * u, x2 * u, y2 * u, p)
        when (kind) {
            "SIREN" -> {
                val path = Path().apply { moveTo(7 * u, 18 * u); lineTo(7 * u, 12 * u); arcTo(7 * u, 7 * u, 17 * u, 17 * u, 180f, 180f, false); lineTo(17 * u, 18 * u) }
                c.drawPath(path, p); c.drawRect(5 * u, 18 * u, 19 * u, 21 * u, p)
                line(12f, 3f, 12f, 5f); line(4.2f, 6.2f, 5.6f, 7.6f); line(19.8f, 6.2f, 18.4f, 7.6f); line(2f, 12f, 4f, 12f); line(20f, 12f, 22f, 12f)
            }
            "HORN" -> {
                c.drawPath(Path().apply { moveTo(3 * u, 10 * u); lineTo(3 * u, 14 * u); lineTo(7 * u, 14 * u); lineTo(13 * u, 18 * u); lineTo(13 * u, 6 * u); lineTo(7 * u, 10 * u); close() }, p)
                c.drawArc(11.5f * u, 8.5f * u, 18.5f * u, 15.5f * u, -50f, 100f, false, p)
                c.drawArc(10f * u, 5f * u, 22f * u, 19f * u, -50f, 100f, false, p)
            }
            "BABY" -> {
                c.drawCircle(12 * u, 11 * u, 7 * u, p)
                line(9f, 10f, 9.01f, 10f); line(15f, 10f, 15.01f, 10f)
                c.drawArc(9 * u, 12 * u, 15 * u, 17 * u, 200f, 140f, false, p)
                line(12f, 4f, 13.5f, 2.5f)
            }
            else -> { // TURN
                c.drawPath(Path().apply { moveTo(10 * u, 5 * u); lineTo(3 * u, 12 * u); lineTo(10 * u, 19 * u) }, p)
                line(3.5f, 12f, 21f, 12f)
            }
        }
    }
}
