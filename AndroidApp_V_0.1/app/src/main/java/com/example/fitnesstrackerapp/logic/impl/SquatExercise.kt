package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.ExerciseConfig
import com.example.fitnesstrackerapp.logic.FormEvaluator
import com.example.fitnesstrackerapp.logic.RepMetrics
import com.example.fitnesstrackerapp.logic.RepPhaseTracker
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class SquatExercise : Exercise {
    override val name: String = "Squat"
    override val color: Int = Color.YELLOW
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Squat"

    override val repHistory = mutableListOf<RepMetrics>()

    private val config = ExerciseConfig(
        name = "Squat",
        idealMinAngleDeg = 70.0,   // knee bent at the bottom
        idealMaxAngleDeg = 160.0,  // legs extended standing
        idealEccentricMs = 2000L..4000L,
        idealConcentricMs = 500L..2000L,
        eccentricLabel = "descent",
        concentricLabel = "stand"
    )
    private val tracker = RepPhaseTracker(config)
    private val evaluator = FormEvaluator(config)

    // MediaPipe Pose Landmarks (right leg)
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
                RepPhaseTracker.Phase.DESCENDING -> "Sit back, go deeper..."
                RepPhaseTracker.Phase.ASCENDING -> "Drive up!"
                RepPhaseTracker.Phase.AT_TOP -> if (repetitions == 0) "Start Squat" else "Ready for next rep"
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
