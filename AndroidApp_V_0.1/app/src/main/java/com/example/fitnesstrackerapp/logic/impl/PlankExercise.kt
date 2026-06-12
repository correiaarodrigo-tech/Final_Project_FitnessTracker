package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class PlankExercise : Exercise {
    override val name: String = "Plank"
    override val color: Int = Color.BLUE
    override var repetitions: Int = 0
    override var state: String = "REST"
    override var feedback: String = "Get into Plank position"
    override val setupInstruction: String = "Turn SIDE-ON (side profile)"
    override val startPositionHint: String = "Plank position, body straight"
    override val isTimeBased: Boolean = true

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

    private val SHOULDER = 12
    private val HIP = 24
    private val KNEE = 26

    override fun progress(): Int = totalSeconds

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

        // Body roughly straight -> a valid plank hold.
        if (angle in 155.0..185.0) {
            if (!isTracking) {
                isTracking = true
                segmentStartMs = now
                state = "HOLDING"
            }
            val heldMs = accumulatedMs + (now - segmentStartMs)
            totalSeconds = (heldMs / 1000).toInt()
            repetitions = totalSeconds
            feedback = "Hold steady — ${totalSeconds}s"
        } else {
            if (isTracking) {
                // Bank the time held so far, then pause.
                accumulatedMs += now - segmentStartMs
                isTracking = false
                state = "PAUSED"
            }
            totalSeconds = (accumulatedMs / 1000).toInt()
            feedback = "Straighten your back — ${totalSeconds}s held"
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
    }
}
