package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.RepMetrics
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

/**
 * Ideal is a straight line from shoulder to hip to knee (180 degrees).
 * The score drops the more the body sags or pikes away from that.
 */
class PlankExercise : Exercise {
    override val name: String = "Plank"
    override val color: Int = Color.BLUE
    override var repetitions: Int = 0
    override var state: String = "REST"
    override var feedback: String = "Get into Plank position"
    override val setupInstruction: String = "Turn SIDE-ON (side profile)"
    override val startPositionHint: String = "Plank position, body straight"
    override val isTimeBased: Boolean = true
    override val repHistory = mutableListOf<RepMetrics>()

    companion object {
        private const val IDEAL_ANGLE_DEG = 180.0
        // Degrees of sag/pike allowed before the score hits 0.
        private const val MAX_DEVIATION_FOR_ZERO_SCORE = 25.0
        private const val HOLD_WINDOW_MIN = 155.0
        private const val HOLD_WINDOW_MAX = 185.0
    }

    override fun isInStartPosition(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size <= KNEE) return false
        val shoulder = landmarks[SHOULDER]
        val hip = landmarks[HIP]

        // Plank-like: torso closer to horizontal than vertical, body aligned.
        val horizontal = kotlin.math.abs(shoulder.x() - hip.x()) >
            kotlin.math.abs(shoulder.y() - hip.y())

        val bodyAngle = AngleCalculator.calculateAngle(
            shoulder.x(), shoulder.y(),
            hip.x(), hip.y(),
            landmarks[KNEE].x(), landmarks[KNEE].y()
        )
        return horizontal && bodyAngle in 150.0..190.0
    }

    // Accumulated hold time survives brief posture breaks instead of resetting.
    private var accumulatedMs: Long = 0
    private var segmentStartMs: Long = 0
    private var isTracking: Boolean = false
    private var totalSeconds: Int = 0

    // Tracks how good the current hold is, to save as one RepMetrics entry.
    private var segmentMinAngle: Double = Double.MAX_VALUE
    private var segmentMaxAngle: Double = -Double.MAX_VALUE
    private var segmentDeviationSum: Double = 0.0
    private var segmentFrameCount: Int = 0
    private var liveScore: Int = 100

    private val SHOULDER = 12
    private val HIP = 24
    private val KNEE = 26

    override fun progress(): Int = totalSeconds

    /** 0-100: 100 at a perfectly straight body, 0 at [MAX_DEVIATION_FOR_ZERO_SCORE] or beyond. */
    private fun alignmentScore(angle: Double): Int {
        val deviation = kotlin.math.abs(angle - IDEAL_ANGLE_DEG)
        val penalty = ((deviation / MAX_DEVIATION_FOR_ZERO_SCORE) * 100.0).coerceIn(0.0, 100.0)
        return (100.0 - penalty).toInt()
    }

    private fun resetSegmentTracking() {
        segmentMinAngle = Double.MAX_VALUE
        segmentMaxAngle = -Double.MAX_VALUE
        segmentDeviationSum = 0.0
        segmentFrameCount = 0
    }

    private fun finalizeSegment(durationMs: Long) {
        if (segmentFrameCount == 0) return
        val avgDeviation = segmentDeviationSum / segmentFrameCount
        val segmentScore = (100.0 - (avgDeviation / MAX_DEVIATION_FOR_ZERO_SCORE) * 100.0)
            .coerceIn(0.0, 100.0).toInt()
        val notes = if (segmentScore >= 90) listOf("Excelente!") else listOf("Mantém as costas direitas!")
        repHistory.add(
            RepMetrics(
                repNumber = repHistory.size + 1,
                eccentricDurationMs = durationMs, // hold duration for this segment
                concentricDurationMs = 0L,
                minAngleDeg = segmentMinAngle,
                maxAngleDeg = segmentMaxAngle,
                formScore = segmentScore,
                feedback = notes
            )
        )
        resetSegmentTracking()
    }

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (landmarks.size <= KNEE) return Triple(repetitions, state, feedback)

        val shoulder = landmarks[SHOULDER]
        val hip = landmarks[HIP]
        val knee = landmarks[KNEE]

        val angle = AngleCalculator.calculateAngle(
            shoulder.x(), shoulder.y(),
            hip.x(), hip.y(),
            knee.x(), knee.y()
        )

        val now = System.currentTimeMillis()
        liveScore = alignmentScore(angle)

        // Body roughly straight -> a valid plank hold.
        if (angle in HOLD_WINDOW_MIN..HOLD_WINDOW_MAX) {
            if (!isTracking) {
                isTracking = true
                segmentStartMs = now
                state = "HOLDING"
            }
            if (angle < segmentMinAngle) segmentMinAngle = angle
            if (angle > segmentMaxAngle) segmentMaxAngle = angle
            segmentDeviationSum += kotlin.math.abs(angle - IDEAL_ANGLE_DEG)
            segmentFrameCount++

            val heldMs = accumulatedMs + (now - segmentStartMs)
            totalSeconds = (heldMs / 1000).toInt()
            repetitions = totalSeconds
            feedback = "Hold steady — ${totalSeconds}s • Score: $liveScore/100"
        } else {
            if (isTracking) {
                // Bank the time held so far, finalise the segment's score, then pause.
                val segmentDurationMs = now - segmentStartMs
                accumulatedMs += segmentDurationMs
                finalizeSegment(segmentDurationMs)
                isTracking = false
                state = "PAUSED"
            }
            totalSeconds = (accumulatedMs / 1000).toInt()
            feedback = "Straighten your back — ${totalSeconds}s held • Score: $liveScore/100"
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        repetitions = 0
        state = "REST"
        feedback = "Get into Plank position"
        accumulatedMs = 0
        segmentStartMs = 0
        isTracking = false
        totalSeconds = 0
        liveScore = 100
        repHistory.clear()
        resetSegmentTracking()
    }
}
