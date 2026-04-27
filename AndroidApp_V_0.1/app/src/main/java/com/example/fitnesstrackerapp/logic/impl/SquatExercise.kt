package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class SquatExercise : Exercise {
    override val name: String = "Squat"
    override val color: Int = Color.YELLOW
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Squat"

    private var isDown: Boolean = false
    
    // MediaPipe Pose Landmarks
    private val HIP = 24
    private val KNEE = 26
    private val ANKLE = 28

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (landmarks.size <= ANKLE) return Triple(repetitions, state, feedback)

        val hip = landmarks[HIP]
        val knee = landmarks[KNEE]
        val ankle = landmarks[ANKLE]

        val angle = AngleCalculator.calculateAngle(
            hip.x(), hip.y(),
            knee.x(), knee.y(),
            ankle.x(), ankle.y()
        )

        if (angle < 70 && !isDown) {
            isDown = true
            state = "DOWN"
            feedback = "Good, now go UP!"
        } else if (angle > 160 && isDown) {
            isDown = false
            repetitions++
            state = "UP"
            feedback = "Repetition complete!"
        } else if (angle > 160 && !isDown) {
            feedback = "Go down to squat"
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        repetitions = 0
        state = "UP"
        feedback = "Reset"
        isDown = false
    }
}
