package com.example.fitnesstrackerapp.logic

/**
 * Simple "lower then lift" state machine for counting reps and timing them.
 *
 * AT_TOP -> DESCENDING -> ASCENDING -> AT_TOP (rep complete).
 * A rep counts once it passes countThreshold, even if it doesn't reach the
 * ideal depth - it'll just score lower instead of being thrown away.
 */
class RepPhaseTracker(private val config: ExerciseConfig) {

    enum class Phase { AT_TOP, DESCENDING, ASCENDING }

    /** Raw timing + range-of-motion output for one completed rep. */
    data class RepCycle(
        val eccentricDurationMs: Long,
        val concentricDurationMs: Long,
        val minAngleDeg: Double,
        val maxAngleDeg: Double
    )

    var phase: Phase = Phase.AT_TOP
        private set

    private var eccentricStartMs = 0L
    private var concentricStartMs = 0L
    private var eccentricMs = 0L
    private var minAngle = Double.MAX_VALUE
    private var maxAngle = -Double.MAX_VALUE

    // Bottom / top of the movement (rep counting thresholds).
    private val bottomThreshold get() = config.countThresholdDeg
    private val topThreshold get() = config.idealMaxAngleDeg
    // The user has clearly started moving once they leave the top zone.
    private val descentOnset get() = config.idealMaxAngleDeg - config.minAngleTolerance

    /**
     * Feeds the latest joint [angle]. Returns a [RepCycle] on the frame a full
     * rep completes, otherwise null.
     */
    fun update(angle: Double, nowMs: Long = System.currentTimeMillis()): RepCycle? {
        when (phase) {
            Phase.AT_TOP -> {
                if (angle < descentOnset) {
                    phase = Phase.DESCENDING
                    eccentricStartMs = nowMs
                    minAngle = angle
                    maxAngle = angle
                }
            }

            Phase.DESCENDING -> {
                if (angle < minAngle) minAngle = angle
                when {
                    angle <= bottomThreshold -> {
                        // Reached depth: eccentric finished, concentric begins.
                        eccentricMs = nowMs - eccentricStartMs
                        concentricStartMs = nowMs
                        phase = Phase.ASCENDING
                    }
                    angle >= topThreshold -> {
                        // Returned to the top without reaching depth: partial rep, discard.
                        phase = Phase.AT_TOP
                    }
                }
            }

            Phase.ASCENDING -> {
                if (angle < minAngle) minAngle = angle
                if (angle > maxAngle) maxAngle = angle
                if (angle >= topThreshold) {
                    val cycle = RepCycle(
                        eccentricDurationMs = eccentricMs,
                        concentricDurationMs = nowMs - concentricStartMs,
                        minAngleDeg = minAngle,
                        maxAngleDeg = maxAngle
                    )
                    phase = Phase.AT_TOP
                    return cycle
                }
            }
        }
        return null
    }

    fun reset() {
        phase = Phase.AT_TOP
        eccentricStartMs = 0L
        concentricStartMs = 0L
        eccentricMs = 0L
        minAngle = Double.MAX_VALUE
        maxAngle = -Double.MAX_VALUE
    }
}
