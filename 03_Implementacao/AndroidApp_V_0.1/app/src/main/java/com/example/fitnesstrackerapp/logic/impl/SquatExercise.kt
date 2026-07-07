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
    override val setupInstruction: String = "Turn SIDE-ON to the camera"
    override val startPositionHint: String = "Stand tall, legs straight"

    override val repHistory = mutableListOf<RepMetrics>()

    private val config = ExerciseConfig(
        name = "Squat",
        idealMinAngleDeg = 70.0,   // knee bent at the bottom
        idealMaxAngleDeg = 160.0,  // legs extended standing
        countThresholdDeg = 90.0,  // below this it's not even a real attempt
        idealEccentricMs = 2000L..4000L,
        idealConcentricMs = 500L..2000L,
        eccentricLabel = "descent",
        concentricLabel = "stand"
    )
    private val tracker = RepPhaseTracker(config)
    private val evaluator = FormEvaluator(config)

    // MediaPipe Pose Landmarks (right leg + shoulder for body orientation)
    private val SHOULDER = 12
    private val HIP = 24
    private val KNEE = 26
    private val ANKLE = 28

    override fun isInStartPosition(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size <= ANKLE) return false
        val shoulder = landmarks[SHOULDER]
        val hip = landmarks[HIP]

        // Standing upright: torso closer to vertical than horizontal.
        val vertical = kotlin.math.abs(shoulder.y() - hip.y()) >
            kotlin.math.abs(shoulder.x() - hip.x())

        val kneeAngle = AngleCalculator.calculateAngle(
            hip.x(), hip.y(),
            landmarks[KNEE].x(), landmarks[KNEE].y(),
            landmarks[ANKLE].x(), landmarks[ANKLE].y()
        )
        return vertical && kneeAngle > 150.0
    }

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
                RepPhaseTracker.Phase.DESCENDING -> "Desce!"
                RepPhaseTracker.Phase.ASCENDING -> "Sobe!"
                RepPhaseTracker.Phase.AT_TOP -> if (repetitions == 0) "Agacha!" else "Outra rep!"
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
