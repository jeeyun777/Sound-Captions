package com.soundwitness.app.vehicle

import android.annotation.SuppressLint
import android.content.Context
import android.os.SystemClock
import android.util.Log
import ai.pleos.playground.vehicle.Vehicle
import ai.pleos.playground.vehicle.api.model.area.SeatArea
import ai.pleos.playground.vehicle.api.model.area.WindowArea
import ai.pleos.playground.vehicle.constant.area.SeatValue
import ai.pleos.playground.vehicle.constant.area.TurnSignalValue
import ai.pleos.playground.vehicle.api.model.capability.AccessType
import ai.pleos.playground.vehicle.constant.control.CruiseControl
import ai.pleos.playground.vehicle.constant.control.OilLevelState
import ai.pleos.playground.vehicle.constant.control.VehicleGear
import ai.pleos.playground.vehicle.listener.BrakePedalStatusListener
import ai.pleos.playground.vehicle.listener.CruiseControlStateListener
import ai.pleos.playground.vehicle.listener.DistanceDrivenListener
import ai.pleos.playground.vehicle.listener.GearSelectionListener
import ai.pleos.playground.vehicle.listener.HvacFanSpeedListener
import ai.pleos.playground.vehicle.listener.OilLevelStateListener
import ai.pleos.playground.vehicle.listener.SeatOccupancyListener
import ai.pleos.playground.vehicle.listener.SteeringWheelAngleListener
import ai.pleos.playground.vehicle.listener.TurnSignalStateListener
import ai.pleos.playground.vehicle.listener.WindowPositionListener
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Collects the vehicle signals SoundWitness needs and keeps the latest values.
 * Every signal is checked with its checkXXXCapabilities call first; a signal the
 * vehicle does not support stays null and is shown as "미지원" in reports.
 */
@SuppressLint("MissingPermission")
class VehicleMonitor(context: Context) {

    private val tag = "VehicleMonitor"
    private val vehicle = Vehicle(context.applicationContext)

    @Volatile var brakePressed: Boolean? = null; private set
    @Volatile var steeringAngle: Float? = null; private set
    @Volatile var gear: VehicleGear? = null; private set
    @Volatile var cruiseOn: Boolean? = null; private set
    @Volatile var fanSpeed: Int? = null; private set
    @Volatile var fanMax: Int = 8; private set
    @Volatile var windowOpenPercent: Int? = null; private set
    @Volatile var windowMax: Int = 100; private set
    @Volatile var brakeFluidLow: Boolean? = null; private set
    @Volatile var engineOilLow: Boolean? = null; private set
    @Volatile var turnSignal: TurnSignalValue? = null; private set
    private val rearSeats = java.util.concurrent.ConcurrentHashMap<String, Boolean>()
    /** someone sits in the 2nd/3rd row (null when the vehicle does not report it) */
    val rearOccupied: Boolean? get() = if (rearSeats.isEmpty()) null else rearSeats.values.any { it }

    /** speed estimated from odometer changes, km/h */
    @Volatile var speedKmh: Float? = null; private set
    private val odoSamples = ArrayDeque<Pair<Long, Float>>()

    val supported = mutableMapOf<String, Boolean>()
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    fun addChangeListener(l: () -> Unit) { listeners += l }
    private fun changed() = listeners.forEach { it() }

    fun start() {
        vehicle.initialize()
        startBrake(); startSteering(); startGear(); startCruise()
        startOdometer(); startFan(); startWindow(); startSafety()
        startTurnSignal(); startRearSeats()
    }

    fun stop() {
        runCatching { vehicle.release() }
    }

    fun snapshot(): VehicleSnapshot = VehicleSnapshot(
        timeMs = System.currentTimeMillis(),
        brake = brakePressed,
        steeringDeg = steeringAngle,
        speedKmh = speedKmh,
        cruiseOn = cruiseOn,
        fanLevel = fanSpeed?.let { it.toFloat() / fanMax.coerceAtLeast(1) },
        windowOpen = windowOpenPercent?.let { it > windowMax * 0.1f },
        brakeFluidLow = brakeFluidLow,
        engineOilLow = engineOilLow,
        gear = gear?.name,
        turnSignal = turnSignal?.name,
        rearOccupied = rearOccupied,
    )

    // ---------- each signal ----------

    private fun ok(name: String, access: AccessType): Boolean {
        val r = access == AccessType.READ || access == AccessType.READ_WRITE
        supported[name] = r
        if (!r) Log.i(tag, "$name not readable ($access)")
        return r
    }

    private fun fail(name: String): (Exception) -> Unit = { e ->
        supported[name] = false
        Log.w(tag, "$name failed: ${e.message}")
    }

    private fun startBrake() {
        val b = vehicle.getBrake()
        b.checkBrakePedalStatusCapabilities({ cap ->
            if (!ok("brake", cap.accessType)) return@checkBrakePedalStatusCapabilities
            b.getCurrentBrakePedalStatus({ brakePressed = it; changed() }, fail("brake"))
            b.registerBrakePedalStatus(object : BrakePedalStatusListener {
                override fun onBrakePedalStatusUpdated(status: Boolean?) { brakePressed = status; changed() }
                override fun onFailed(e: Exception) { fail("brake")(e) }
            })
        }, fail("brake"))
    }

    private fun startSteering() {
        val s = vehicle.getSteeringWheel()
        s.checkSteeringWheelAngleCapabilities({ cap, _ ->
            if (!ok("steering", cap.accessType)) return@checkSteeringWheelAngleCapabilities
            s.getCurrentSteeringWheelAngle({ steeringAngle = it; changed() }, fail("steering"))
            s.registerSteeringWheelAngle(object : SteeringWheelAngleListener {
                override fun onSteeringWheelAngleUpdated(angle: Float?) { steeringAngle = angle; changed() }
                override fun onFailed(e: Exception) { fail("steering")(e) }
            })
        }, fail("steering"))
    }

    @Suppress("DEPRECATION")
    private fun startGear() {
        val g = vehicle.getGear()
        runCatching {
            g.getCurrentGearSelectionState({ gear = it; supported["gear"] = true; changed() }, fail("gear"))
            g.registerGearSelection(object : GearSelectionListener {
                override fun onSelected(gear: VehicleGear?) { this@VehicleMonitor.gear = gear; changed() }
                override fun onFailed(e: Exception) { fail("gear")(e) }
            })
        }.onFailure { fail("gear")(it as? Exception ?: Exception(it)) }
    }

    private fun startCruise() {
        val d = vehicle.getDrivingMode()
        d.checkCruiseControlStateCapabilities({ cap ->
            if (!ok("cruise", cap.accessType)) return@checkCruiseControlStateCapabilities
            d.getCruiseControlState({ cruiseOn = it?.isActive(); changed() }, fail("cruise"))
            d.registerCruiseControlState(object : CruiseControlStateListener {
                override fun onCruiseControlStateUpdated(state: CruiseControl?) { cruiseOn = state?.isActive(); changed() }
                override fun onFailed(e: Exception) { fail("cruise")(e) }
            })
        }, fail("cruise"))
    }

    private fun CruiseControl.isActive(): Boolean = this == CruiseControl.ACTIVATED

    private fun startOdometer() {
        val o = vehicle.getOdometer()
        o.checkCurrentDistanceDrivenCapabilities({ cap ->
            if (!ok("odometer", cap.accessType)) return@checkCurrentDistanceDrivenCapabilities
            o.getCurrentDistanceDriven({ it?.let(::onOdometer) }, fail("odometer"))
            o.registerDistanceDriven(object : DistanceDrivenListener {
                override fun onDistanceDrivenUpdated(value: Float?) { value?.let(::onOdometer) }
                override fun onFailed(e: Exception) { fail("odometer")(e) }
            })
        }, fail("odometer"))
    }

    /** speed = distance change over the last few seconds of odometer updates */
    @Synchronized
    private fun onOdometer(km: Float) {
        val now = SystemClock.elapsedRealtime()
        odoSamples.addLast(now to km)
        while (odoSamples.size > 2 && now - odoSamples.first().first > 5000) odoSamples.removeFirst()
        if (odoSamples.size >= 2) {
            val (t0, k0) = odoSamples.first()
            val dt = (now - t0) / 3_600_000f
            if (dt > 0f) speedKmh = ((km - k0) / dt).coerceIn(0f, 300f)
        }
        changed()
    }

    /** called once a second so speed drops to 0 when the odometer stops changing */
    @Synchronized
    fun tick() {
        val last = odoSamples.lastOrNull() ?: return
        if (SystemClock.elapsedRealtime() - last.first > 4000) { speedKmh = 0f; changed() }
    }

    private fun startFan() {
        val h = vehicle.getHVAC()
        h.checkHvacFanSpeedCapabilities({ cap, bounds ->
            if (!ok("fan", cap.accessType) || bounds.isEmpty()) return@checkHvacFanSpeedCapabilities
            val (area, b) = bounds.entries.first()
            fanMax = b.maxValue
            h.getHvacFanSpeed(area, { fanSpeed = it; changed() }, fail("fan"))
            h.registerHvacFanSpeed(area, object : HvacFanSpeedListener {
                override fun onHvacFanSpeedUpdated(speed: Int?) { fanSpeed = speed; changed() }
                override fun onFailed(e: Exception) { fail("fan")(e) }
            })
        }, fail("fan"))
    }

    private fun startWindow() {
        val w = vehicle.getWindow()
        w.checkWindowPositionCapabilities({ cap, bounds ->
            if (!ok("window", cap.accessType) || bounds.isEmpty()) return@checkWindowPositionCapabilities
            val (area, b) = bounds.entries.first()
            windowMax = b.maxValue
            w.getCurrentWindowPosition(area, { windowOpenPercent = it; changed() }, fail("window"))
            w.registerWindowPosition(area, object : WindowPositionListener {
                override fun onWindowPositionUpdated(position: Int?) { windowOpenPercent = position; changed() }
                override fun onFailed(e: Exception) { fail("window")(e) }
            })
        }, fail("window"))
    }


    private fun startSafety() {
        val s = vehicle.getSafety()
        val isLow: (OilLevelState?) -> Boolean? = { it?.let { st -> st == OilLevelState.LOW || st == OilLevelState.CRITICALLY_LOW } }
        runCatching {
            s.getBrakeFluidLevelState({ brakeFluidLow = isLow(it); changed() }, fail("brakeFluid"))
            s.registerBrakeFluidLevelState(object : OilLevelStateListener {
                override fun onOilLevelUpdated(level: OilLevelState?) { brakeFluidLow = isLow(level); changed() }
                override fun onFailed(e: Exception) { fail("brakeFluid")(e) }
            })
            s.getEngineOilLevelState({ engineOilLow = isLow(it); changed() }, fail("engineOil"))
        }
    }

    private fun startTurnSignal() {
        val t = vehicle.getTurnSignal()
        t.checkTurnSignalStateCapabilities({ cap ->
            if (!ok("turnSignal", cap.accessType)) return@checkTurnSignalStateCapabilities
            t.getCurrentTurnSignalState({ turnSignal = it; changed() }, fail("turnSignal"))
            t.registerTurnSignalState(object : TurnSignalStateListener {
                override fun onTurnSignalStateUpdated(state: TurnSignalValue?) { turnSignal = state; changed() }
                override fun onFailed(e: Exception) { fail("turnSignal")(e) }
            })
        }, fail("turnSignal"))
    }

    /** rear-seat occupancy turns on the crying-child feature */
    private fun startRearSeats() {
        val s = vehicle.getSeat()
        s.checkSeatOccupancyCapabilities({ cap, areas ->
            if (!ok("rearSeat", cap.accessType)) return@checkSeatOccupancyCapabilities
            areas.filter { a -> a.values.any { it.name.startsWith("ROW_2") || it.name.startsWith("ROW_3") } }.forEach { area ->
                val key = area.values.joinToString { it.name }
                s.getCurrentSeatOccupancy(area, { rearSeats[key] = it == true; changed() }, fail("rearSeat"))
                s.registerSeatOccupancy(area, object : SeatOccupancyListener {
                    override fun onSeatOccupancyUpdated(occupancy: Boolean?) { rearSeats[key] = occupancy == true; changed() }
                    override fun onFailed(e: Exception) { fail("rearSeat")(e) }
                })
            }
        }, fail("rearSeat"))
    }

    @Suppress("unused") private val keepSeat = SeatValue.ROW_2_LEFT
    @Suppress("unused") private val keepTypes = listOf(SeatArea::class, WindowArea::class)
}

data class VehicleSnapshot(
    val timeMs: Long,
    val brake: Boolean?,
    val steeringDeg: Float?,
    val speedKmh: Float?,
    val cruiseOn: Boolean?,
    val fanLevel: Float?,
    val windowOpen: Boolean?,
    val brakeFluidLow: Boolean?,
    val engineOilLow: Boolean?,
    val gear: String?,
    val turnSignal: String? = null,
    val rearOccupied: Boolean? = null,
)
