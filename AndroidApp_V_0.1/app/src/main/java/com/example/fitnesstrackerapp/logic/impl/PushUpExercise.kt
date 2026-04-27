package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class PushUpExercise : Exercise {
    override val name: String = "Push-Up"
    override val color: Int = Color.CYAN
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Push-Up"

    private var isDown: Boolean = false
    
    // MediaPipe Pose Landmarks
    private val SHOULDER = 12
    private val ELBOW = 14
    private val WRIST = 16

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (landmarks.size <= WRIST) return Triple(repetitions, state, feedback)

        val shoulder = landmarks[SHOULDER]
        val elbow = landmarks[ELBOW]
        val wrist = landmarks[WRIST]

        val angle = AngleCalculator.calculateAngle(
            shoulder.x(), shoulder.y(),
            elbow.x(), elbow.y(),
            wrist.x(), wrist.y()
        )

        if (angle < 70 && !isDown) {
            isDown = true
            state = "DOWN"
            feedback = "Push up!"
        } else if (angle > 150 && isDown) {
            isDown = false
            repetitions++
            state = "UP"
            feedback = "Nice rep!"
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
