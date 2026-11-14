package com.soundwitness.app.eval

import com.soundwitness.app.audio.SAMPLE_RATE
import com.soundwitness.app.events.EventDetector
import com.soundwitness.app.events.EventType
import com.soundwitness.app.learn.CountModel
import com.soundwitness.app.learn.Encoder
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Step-by-step evaluation on real recordings (ESC-50, resampled to 16 kHz by prep_esc50.py).
 *
 * A "car" = one background condition (engine / rain / wind / train) + how much the cabin
 * muffles outside sounds (low-pass). Leave-one-car-out: the factory model never sees the
 * test car. ESC-50 folds: 1-2 factory training, 3 = what the car meets after install
 * (unlabeled background + events confirmed by driver reaction), 4-5 = test.
 *
 *  S0 synthetic factory model (current APK)
 *  S1 factory model from real recordings
 *  S2 + this car's background learned on its own (no labels)
 *  S3 + a few events confirmed by the driver's reaction
 *  S4 wrong labels learned: without vs with the self test rollback
 *  plus: a conventional classifier trained with backprop on the same data
 */
class RealDataLadderTest {

    private val dir = File(System.getProperty("esc50.dir") ?: System.getenv("ESC50_16K") ?: "../data/esc50_16k")

    data class Car(val name: String, val bg: String, val lpHz: Float)
    private val cars = listOf(Car("A 엔진", "engine", 1500f), Car("B 비", "rain", 2500f), Car("C 바람", "wind", 4000f), Car("D 기차", "train", 2000f))
    private val generic = listOf("airplane", "helicopter")
    private val distractors = listOf("dog", "clock_alarm", "church_bells", "laughing", "crow")       // test only
    private val others = listOf("rooster", "cat", "door_wood_knock", "glass_breaking", "chainsaw", "fireworks", "sneezing", "coughing",
        "vacuum_cleaner", "keyboard_typing", "footsteps", "clapping", "chirping_birds", "thunderstorm", "hand_saw", "cow")  // factory only
    private val evCat = mapOf(EventType.SIREN to "siren", EventType.HORN to "car_horn", EventType.BABY to "crying_baby")
    private val outside = setOf(EventType.SIREN, EventType.HORN)
    private val snrs = listOf(6f, 0f, -6f)

    class Clip(val fold: Int, val x: FloatArray)

    private val cache = HashMap<String, List<Clip>>()
    private fun clips(cat: String) = cache.getOrPut(cat) {
        File(dir, cat).listFiles()!!.sortedBy { it.name }.map { f ->
            val b = f.readBytes()
            Clip(f.name.substringBefore('_').toInt(), FloatArray(b.size / 2) { i -> ((b[2 * i].toInt() and 0xff) or (b[2 * i + 1].toInt() shl 8)).toShort() / 32768f })
        }
    }

    private fun rms(x: FloatArray, s: Int = 0, n: Int = x.size): Double { var e = 0.0; for (i in s until s + n) e += x[i] * x[i]; return sqrt(e / n + 1e-12) }
    private fun windows(c: Clip): List<FloatArray> = (0..(c.x.size - SAMPLE_RATE) step SAMPLE_RATE / 2).map { c.x.copyOfRange(it, it + SAMPLE_RATE) }
    /** windows where the event is actually sounding (ESC-50 clips are padded with silence) */
    private fun eventWindows(c: Clip, max: Int = 4): List<FloatArray> {
        val w = windows(c); val r = w.map { rms(it) }; val top = r.max()
        return w.indices.filter { r[it] >= top * 10.0.pow(-10 / 20.0) }.sortedByDescending { r[it] }.take(max).map { w[it] }
    }
    private fun bgWindows(cat: String, folds: Set<Int>) = clips(cat).filter { it.fold in folds }.flatMap { windows(it) }.filter { rms(it) > 1e-4 }

    private fun lowpass(x: FloatArray, fc: Float): FloatArray {
        val a = exp(-2.0 * Math.PI * fc / SAMPLE_RATE).toFloat()
        var y1 = 0f; var y2 = 0f
        return FloatArray(x.size) { i -> y1 = (1 - a) * x[i] + a * y1; y2 = (1 - a) * y1 + a * y2; y2 }
    }
    private fun pcm(x: FloatArray) = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32000).toInt().toShort() }

    private val BG_RMS = 0.05
    /** what the cabin mic hears: car background + (muffled) event at a given SNR, times a mic gain */
    private fun mix(bg: FloatArray, ev: FloatArray?, type: EventType?, car: Car, snr: Float, gainDb: Float = 0f): ShortArray {
        val g = 10.0.pow(gainDb / 20.0).toFloat()
        val b = (BG_RMS / rms(bg)).toFloat()
        val out = FloatArray(SAMPLE_RATE) { bg[it] * b }
        if (ev != null) {
            val e = if (type in outside) lowpass(ev, car.lpHz) else ev
            val k = (BG_RMS * 10.0.pow(snr / 20.0) / rms(e)).toFloat()
            for (i in out.indices) out[i] += e[i] * k
        }
        for (i in out.indices) out[i] *= g
        return pcm(out)
    }

    // ---------- data for one leave-one-car-out split ----------

    inner class Split(val target: Car, val seed: Int) {
        val r = Random(seed)
        val trainCars = cars - target
        private fun pickBg(c: Car, folds: Set<Int>) = bgWindows(c.bg, folds).let { it[r.nextInt(it.size)] }
        // factory training (folds 1-2), mixed into the other cars
        // each training window remembers which background (car) it was mixed into
        val trainEventsTagged: Map<EventType, List<Pair<ShortArray, String>>> = evCat.mapValues { (t, cat) ->
            clips(cat).filter { it.fold <= 2 }.flatMap { eventWindows(it) }.flatMap { w ->
                List(2) { val c = trainCars[r.nextInt(trainCars.size)]; mix(pickBg(c, setOf(1, 2)), w, t, c, snrs[r.nextInt(3)]) to c.bg }
            }
        }
        val trainEvents = trainEventsTagged.mapValues { (_, l) -> l.map { it.first } }
        val trainBgTagged: List<Pair<ShortArray, String>> = (trainCars.flatMap { c -> bgWindows(c.bg, setOf(1, 2)).map { mix(it, null, null, c, 0f) to c.bg } } +
            generic.flatMap { g -> bgWindows(g, setOf(1, 2)).map { mix(it, null, null, cars[0], 0f) to g } })
        val trainBg = trainBgTagged.map { it.first }
        val trainOtherTagged: List<Pair<ShortArray, String>> = others.flatMap { o ->
            clips(o).filter { it.fold <= 2 }.flatMap { eventWindows(it, 2) }.map { w -> val c = trainCars[r.nextInt(trainCars.size)]; mix(pickBg(c, setOf(1, 2)), w, null, c, snrs[r.nextInt(3)]) to c.bg }
        }
        /** usual background level per band for each training condition, and their average (the factory profile) */
        val profiles: Map<String, FloatArray> = trainBgTagged.groupBy({ it.second }, { it.first }).mapValues { (_, l) -> meanProfile(l) }
        val factoryProfile: FloatArray = meanProfile(trainBg)
        // calibration: background + distractors from the training cars only (fold 3)
        val calib: List<ShortArray> = trainCars.flatMap { c -> bgWindows(c.bg, setOf(3)).map { mix(it, null, null, c, 0f) } } +
            distractors.flatMap { d -> clips(d).filter { it.fold == 3 }.flatMap { eventWindows(it, 2) }.map { w -> val c = trainCars[r.nextInt(trainCars.size)]; mix(pickBg(c, setOf(3)), w, null, c, 0f) } }
        // after install in the target car (fold 3): unlabeled background stream, reaction-confirmed events
        val carBgStream: List<ShortArray> = bgWindows(target.bg, setOf(1, 2, 3)).map { mix(it, null, null, target, 0f) }
        val confirmed: Map<EventType, List<ShortArray>> = evCat.mapValues { (t, cat) ->
            clips(cat).filter { it.fold == 3 }.take(5).flatMap { eventWindows(it, 3) }.map { mix(pickBg(target, setOf(3)), it, t, target, snrs[r.nextInt(3)]) }
        }
        val wrong: List<ShortArray> = clips("car_horn").filter { it.fold == 3 }.flatMap { eventWindows(it, 4) }.map { mix(pickBg(target, setOf(3)), it, EventType.HORN, target, 0f) }
        val selfTestEvents: Map<EventType, List<FloatArray>> = evCat.mapValues { (t, cat) ->
            clips(cat).filter { it.fold == 3 }.flatMap { eventWindows(it, 1) }.take(10).map { w -> val e = if (t in outside) lowpass(w, target.lpHz) else w; val k = (BG_RMS / rms(e)).toFloat(); FloatArray(SAMPLE_RATE) { e[it] * k } }
        }
        // test (folds 4-5) in the target car
        fun testEvents(gainDb: Float = 0f): Map<EventType, List<Pair<Float, ShortArray>>> = evCat.mapValues { (t, cat) ->
            val rr = Random(seed * 31 + t.ordinal)
            val bgs = bgWindows(target.bg, setOf(4, 5))
            clips(cat).filter { it.fold >= 4 }.flatMap { eventWindows(it) }.flatMap { w -> snrs.map { s -> s to mix(bgs[rr.nextInt(bgs.size)], w, t, target, s, gainDb) } }
        }
        fun testBg(gainDb: Float = 0f) = bgWindows(target.bg, setOf(4, 5)).map { mix(it, null, null, target, 0f, gainDb) }
        fun testDistract(gainDb: Float = 0f): List<ShortArray> {
            val rr = Random(seed * 17); val bgs = bgWindows(target.bg, setOf(4, 5))
            return distractors.flatMap { d -> clips(d).filter { it.fold >= 4 }.flatMap { eventWindows(it, 2) }.map { mix(bgs[rr.nextInt(bgs.size)], it, null, target, 0f, gainDb) } }
        }
    }

    // ---------- metrics ----------

    data class Score(val recall: Map<EventType, Double>, val recallBySnr: Map<Float, Double>, val faBg: Double, val faDis: Double, val auc: Double, val aucBg: Double, val aucDis: Double)

    /** generic classifier interface so the counting detector and the baseline are scored the same way */
    interface Clf { fun margin(p: ShortArray): Pair<EventType, Double>; val thr: Map<EventType, Double> }

    private fun score(c: Clf, s: Split, gainDb: Float = 0f): Score {
        val ev = s.testEvents(gainDb); val bg = s.testBg(gainDb); val dis = s.testDistract(gainDb)
        fun hit(p: ShortArray, t: EventType?): Boolean { val (b, m) = c.margin(p); return if (t == null) m > c.thr[b]!! else (b == t && m > c.thr[b]!!) }
        val recall = ev.mapValues { (t, l) -> l.count { hit(it.second, t) }.toDouble() / l.size }
        val bySnr = snrs.associateWith { s0 -> ev.flatMap { (t, l) -> l.filter { it.first == s0 }.map { hit(it.second, t) } }.let { h -> h.count { it }.toDouble() / h.size } }
        // detection AUC: any-event margin, events vs (background + distractors)
        val pos = ev.values.flatten().map { c.margin(it.second).second }
        val nb = bg.map { c.margin(it).second }; val nd = dis.map { c.margin(it).second }
        fun auc(neg: List<Double>): Double { var a = 0.0; for (p in pos) for (n in neg) a += if (p > n) 1.0 else if (p == n) 0.5 else 0.0; return a / (pos.size * neg.size) }
        return Score(recall, bySnr, bg.count { hit(it, null) }.toDouble() / bg.size, dis.count { hit(it, null) }.toDouble() / dis.size, auc(nb + nd), auc(nb), auc(nd))
    }

    private fun countingClf(d: EventDetector) = object : Clf {
        override fun margin(p: ShortArray): Pair<EventType, Double> { val w = d.classify(p); return w.best to w.margin }
        override val thr get() = d.thresholds
    }

    /** threshold per type = 99th percentile of that type's margin on calibration (non-event) windows of the training cars */
    private fun calibrate(c: Clf, s: Split): Map<EventType, Double> {
        val ms = s.calib.map { c.margin(it) }
        return EventType.entries.associateWith { t -> ms.filter { it.first == t }.map { it.second }.sorted().let { l -> if (l.isEmpty()) 0.0 else l[(l.size * 0.99).toInt().coerceAtMost(l.size - 1)] } }
    }

    private fun meanProfile(l: List<ShortArray>): FloatArray {
        val p = FloatArray(Encoder.SIZE); for (w in l) { val m = Encoder.encode(w).bandMeanDb; for (b in p.indices) p[b] += m[b] / l.size }; return p
    }

    /** [relative] = encode against the background profile of the car each window was recorded in */
    private fun factory(s: Split, relative: Boolean, withOther: Boolean = false): EventDetector {
        fun enc(p: ShortArray, bg: String) = Encoder.encode(p, ref = if (relative) s.profiles[bg] else null)
        val m = EventType.entries.associateWith { t -> CountModel().also { cm -> s.trainEventsTagged[t]!!.forEach { (p, bg) -> cm.add(enc(p, bg)) } } }
        val bg = CountModel().also { cm -> s.trainBgTagged.forEach { (p, g) -> cm.add(enc(p, g)) } }
        val prof = if (relative) s.factoryProfile else null
        val oth = if (withOther) CountModel().also { cm -> s.trainOtherTagged.forEach { (p, g) -> cm.add(enc(p, g)) } } else null
        val d = EventDetector(); d.installFactory(m, bg, null, prof, oth)
        d.installFactory(m, bg, calibrate(countingClf(d), s), prof, oth)
        return d
    }

    // ---------- conventional baseline: softmax regression on the same 28x28 log-band image, SGD (backprop) ----------

    inner class Softmax(val s: Split) : Clf {
        val K = 4; val D = Encoder.SIZE * Encoder.SIZE
        val w = Array(K) { DoubleArray(D) }; val b = DoubleArray(K)
        var mean = DoubleArray(D); var std = DoubleArray(D) { 1.0 }
        var macs = 0L
        override var thr: Map<EventType, Double> = emptyMap()
        fun feat(p: ShortArray): DoubleArray { val img = Encoder.encode(p).image; val f = DoubleArray(D); var i = 0; for (row in img) for (v in row) f[i++] = v.toDouble(); return f }
        fun norm(f: DoubleArray) = DoubleArray(D) { (f[it] - mean[it]) / std[it] }
        fun logits(x: DoubleArray) = DoubleArray(K) { k -> var z = b[k]; for (j in 0 until D) z += w[k][j] * x[j]; z }
        fun fitNorm(data: List<DoubleArray>) {
            mean = DoubleArray(D) { j -> data.sumOf { it[j] } / data.size }
            std = DoubleArray(D) { j -> sqrt(data.sumOf { (it[j] - mean[j]).pow(2) } / data.size).coerceAtLeast(1e-3) }
        }
        fun train(xs: List<DoubleArray>, ys: List<Int>, epochs: Int, lr: Double) {
            val rr = Random(5); val idx = xs.indices.toMutableList()
            val cw = DoubleArray(K) { k -> ys.size.toDouble() / (K * ys.count { it == k }.coerceAtLeast(1)) }   // class balance
            repeat(epochs) {
                idx.shuffle(rr)
                for (i in idx) {
                    val x = xs[i]; val z = logits(x); val mz = z.max(); val e = z.map { exp(it - mz) }; val sum = e.sum()
                    for (k in 0 until K) { val g = cw[ys[i]] * (e[k] / sum - if (k == ys[i]) 1.0 else 0.0); b[k] -= lr * g; for (j in 0 until D) w[k][j] -= lr * (g * x[j] + 1e-4 * w[k][j]) }
                    macs += 2L * K * D
                }
            }
        }
        override fun margin(p: ShortArray): Pair<EventType, Double> {
            val z = logits(norm(feat(p)))
            var best = 0; for (k in 0 until 3) if (z[k] > z[best]) best = k
            return EventType.entries[best] to (z[best] - z[3])
        }
        fun factoryTrain() {
            val raw = EventType.entries.flatMap { t -> s.trainEvents[t]!!.map { feat(it) to t.ordinal } } + s.trainBg.map { feat(it) to 3 }
            fitNorm(raw.map { it.first })
            train(raw.map { norm(it.first) }, raw.map { it.second }, 15, 0.01)
            thr = calibrate(this, s)
        }
        fun adaptData() = EventType.entries.flatMap { t -> s.confirmed[t]!!.map { norm(feat(it)) to t.ordinal } } + s.carBgStream.map { norm(feat(it)) to 3 }
        fun adapt(raw: List<Pair<DoubleArray, Int>>) = train(raw.map { it.first }, raw.map { it.second }, 5, 0.005)
    }

    private fun fmt(sc: Score) = "%s | 오경보 배경 %4.1f%% 다른소리 %4.1f%% | AUC 배경 %.3f 다른소리 %.3f".format(
        EventType.entries.joinToString(" ") { "%s %3.0f%%".format(it.label.take(4), 100 * sc.recall[it]!!) }, 100 * sc.faBg, 100 * sc.faDis, sc.aucBg, sc.aucDis)

    private fun avg(l: List<Score>) = Score(
        EventType.entries.associateWith { t -> l.map { it.recall[t]!! }.average() },
        snrs.associateWith { s -> l.map { it.recallBySnr[s]!! }.average() },
        l.map { it.faBg }.average(), l.map { it.faDis }.average(), l.map { it.auc }.average(), l.map { it.aucBg }.average(), l.map { it.aucDis }.average())

    @Test
    fun ladder() {
        assumeTrue("ESC-50 not prepared at $dir", File(dir, "siren").isDirectory)
        val rows = linkedMapOf<String, MutableList<Score>>()
        fun put(k: String, s: Score) { rows.getOrPut(k) { mutableListOf() } += s }
        var countAddOps = 0L; var countAddMs = 0L; var smAdaptMacs = 0L; var smAdaptMs = 0L; var rolled = 0

        for ((i, car) in cars.withIndex()) {
            val s = Split(car, 100 + i)
            // S0 synthetic factory model (current APK), thresholds as shipped
            val syn = EventDetector().apply { trainPrototypes() }
            put("S0 합성 소리 출고 모델(현재 APK)", score(countingClf(syn), s))
            // S1 real-data factory model, plain encoding (loudest cells)
            val d0 = factory(s, relative = false)
            put("S1 실제 소리 출고 모델(배경 기준 없이)", score(countingClf(d0), s))
            // S1b cells compared with the background first (factory = average background of the training cars)
            put("S1b + 배경 기준선을 뺀 뒤 카운팅", score(countingClf(factory(s, relative = true)), s))
            val d = factory(s, relative = true, withOther = true)
            put("S1c + '기타 소리' 모델", score(countingClf(d), s))
            // S2 learn this car's background on its own (same gate as the app: only windows judged "no event")
            val bgEnc = ArrayList<Pair<ShortArray, EventDetector.Window>>()
            for (p in s.carBgStream) { val w = d.classify(p); if (w.type == null) bgEnc += p to w }
            var t0 = System.nanoTime(); for ((p, w) in bgEnc) d.learnBackground(p, w); countAddMs += (System.nanoTime() - t0) / 1_000_000
            d.recalibrate()
            put("S2 + 이 차 배경 소음 자동 적응", score(countingClf(d), s))
            put("S2 + 마이크 감도 +12dB", score(countingClf(d), s, 12f))
            put("S2 + 마이크 감도 -12dB", score(countingClf(d), s, -12f))
            // S3 events confirmed by driver reaction
            val conf = s.confirmed.mapValues { (_, l) -> l.map { d.classify(it).enc } }
            t0 = System.nanoTime(); for ((t, l) in conf) d.learnEvent(t, l); countAddMs += (System.nanoTime() - t0) / 1_000_000
            countAddOps += (bgEnc.size + conf.values.sumOf { it.size }) * 676L
            put("S3 + 운전자 반응으로 확인된 사건음 학습", score(countingClf(d), s))
            // S4 wrong labels: horns learned as sirens. Self test uses real test sounds + this car's background.
            d.testEvents = s.selfTestEvents
            d.selfTest()                                                // baseline = S3 model
            val wrongEnc = s.wrong.map { d.classify(it).enc }
            val poisoned = EventDetector().also { it.installFactory(d.models, d.background, d.thresholds, d.profile, d.other) }
            poisoned.learnEvent(EventType.SIREN, wrongEnc); poisoned.learnEvent(EventType.SIREN, wrongEnc)
            put("S4 잘못 학습(되돌림 없음)", score(countingClf(poisoned), s))
            d.learnEvent(EventType.SIREN, wrongEnc); d.learnEvent(EventType.SIREN, wrongEnc)
            val tr = d.selfTest()
            if (tr.rolledBack) rolled++
            put("S4 잘못 학습 → 자가 시험 후", score(countingClf(d), s))

            // conventional baseline
            val sm = Softmax(s); sm.factoryTrain()
            put("B1 기존 방식 출고 모델(역전파 학습)", score(sm, s))
            put("B1 + 마이크 감도 +12dB", score(sm, s, 12f))
            put("B1 + 마이크 감도 -12dB", score(sm, s, -12f))
            val adaptData = sm.adaptData()
            val m0 = sm.macs; val t1 = System.nanoTime(); sm.adapt(adaptData); smAdaptMs += (System.nanoTime() - t1) / 1_000_000; smAdaptMacs += sm.macs - m0
            put("B3 기존 방식 + 같은 데이터로 재학습", score(sm, s))
            println("car ${car.name}: S3 ${fmt(rows["S3 + 운전자 반응으로 확인된 사건음 학습"]!!.last())}")
        }
        println("\n=== 평균 (4개 차 조건, 각 차는 학습에 쓰지 않고 평가) ===")
        for ((k, l) in rows) println("%-34s %s | SNR +6/0/-6 재현율 %s".format(k, fmt(avg(l)), avg(l).recallBySnr.values.joinToString("/") { "%.0f".format(100 * it) }))
        println("\n=== 차 안 학습 비용 (4개 차 합계) ===")
        println("카운팅: 정수 더하기 %,d회, %d ms (역전파 없음)".format(countAddOps, countAddMs))
        println("기존 방식 재학습: 곱셈·덧셈 %,d회, %d ms (5 epoch 역전파)".format(smAdaptMacs, smAdaptMs))
        println("자가 시험 되돌림 발동: $rolled / ${cars.size} 차")
    }
}
