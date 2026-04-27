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

    private var startTime: Long = 0
    private var totalSeconds: Int = 0
    private var isTracking: Boolean = false

    private val SHOULDER = 12
    private val HIP = 24
    private val KNEE = 26

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

        // Range: 155 to 185
        if (angle in 155.0..185.0) {
            if (!isTracking) {
                isTracking = true
                startTime = System.currentTimeMillis()
                state = "HOLDING"
            } else {
                val currentTrackingTime = (System.currentTimeMillis() - startTime) / 1000
                totalSeconds = currentTrackingTime.toInt()
                repetitions = totalSeconds / 10
                feedback = "Holding Plank: ${totalSeconds}s"
            }
        } else {
            if (isTracking) {
                isTracking = false
                // Pause logic or just stop tracking
                state = "PAUSED"
                feedback = "Fix your posture! Total: ${totalSeconds}s"
            } else {
                feedback = "Align your back"
            }
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        repetitions = 0
        state = "REST"
        feedback = "Reset"
        startTime = 0
        totalSeconds = 0
        isTracking = false
    }
}
