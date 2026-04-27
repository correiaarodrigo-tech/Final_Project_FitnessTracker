package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.Exercise
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class RestExercise(private val durationSeconds: Int) : Exercise {
    override val name: String = "REST"
    override val color: Int = Color.GRAY
    override var repetitions: Int = 0
    override var state: String = "RESTING"
    override var feedback: String = "Rest for ${durationSeconds}s"

    private var startTime: Long = 0
    private var isTracking: Boolean = false

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (!isTracking) {
            isTracking = true
            startTime = System.currentTimeMillis()
        }

        val elapsed = (System.currentTimeMillis() - startTime) / 1000
        val remaining = durationSeconds - elapsed.toInt()

        if (remaining > 0) {
            repetitions = remaining
            feedback = "Remaining: ${remaining}s"
        } else {
            repetitions = 0
            feedback = "Rest finished!"
            state = "FINISHED"
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        startTime = 0
        isTracking = false
        repetitions = 0
        state = "RESTING"
    }
}
