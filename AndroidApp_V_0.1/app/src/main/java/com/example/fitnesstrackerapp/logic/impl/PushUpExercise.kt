package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.ExerciseConfig
import com.example.fitnesstrackerapp.logic.FormEvaluator
import com.example.fitnesstrackerapp.logic.RepMetrics
import com.example.fitnesstrackerapp.logic.RepPhaseTracker
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class PushUpExercise : Exercise {
    override val name: String = "Push-Up"
    override val color: Int = Color.CYAN
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Push-Up"

    override val repHistory = mutableListOf<RepMetrics>()

    private val config = ExerciseConfig(
        name = "Push-Up",
        idealMinAngleDeg = 70.0,   // elbow bent at the bottom
        idealMaxAngleDeg = 150.0,  // arms extended at the top
        idealEccentricMs = 1500L..3000L,
        idealConcentricMs = 500L..2000L,
        eccentricLabel = "descent",
        concentricLabel = "press"
    )
    private val tracker = RepPhaseTracker(config)
    private val evaluator = FormEvaluator(config)

    // MediaPipe Pose Landmarks (right arm)
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

        val cycle = tracker.update(angle)
        state = if (tracker.phase == RepPhaseTracker.Phase.DESCENDING) "DOWN" else "UP"

        if (cycle != null) {
            repetitions++
            val (score, notes) = evaluator.evaluate(cycle)
            repHistory.add(
                RepMetrics(
                    repNumber = repetitions,
                    eccentricDurationMs = cycle.eccentricDurationMs,
                    concentricDurationMs = cycle.concentricDurationMs,
                    minAngleDeg = cycle.minAngleDeg,
                    maxAngleDeg = cycle.maxAngleDeg,
                    formScore = score,
                    feedback = notes
                )
            )
            feedback = "Rep $repetitions • $score/100 — ${notes.first()}"
        } else {
            feedback = when (tracker.phase) {
                RepPhaseTracker.Phase.DESCENDING -> "Lower under control..."
                RepPhaseTracker.Phase.ASCENDING -> "Push up!"
                RepPhaseTracker.Phase.AT_TOP -> if (repetitions == 0) "Start Push-Up" else "Ready for next rep"
            }
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        repetitions = 0
        state = "UP"
        feedback = "Reset"
        repHistory.clear()
        tracker.reset()
    }
}
