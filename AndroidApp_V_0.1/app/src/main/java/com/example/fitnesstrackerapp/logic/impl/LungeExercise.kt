package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.ExerciseConfig
import com.example.fitnesstrackerapp.logic.FormEvaluator
import com.example.fitnesstrackerapp.logic.RepMetrics
import com.example.fitnesstrackerapp.logic.RepPhaseTracker
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class LungeExercise : Exercise {
    override val name: String = "Lunge"
    override val color: Int = Color.MAGENTA
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Lunge (Alternate Legs)"
    override val setupInstruction: String = "FACE the camera"
    override val startPositionHint: String = "Stand tall, feet together"

    override val repHistory = mutableListOf<RepMetrics>()

    private var lastForwardLeg: String = "" // "LEFT" or "RIGHT"

    private val config = ExerciseConfig(
        name = "Lunge",
        idealMinAngleDeg = 80.0,   // back knee bent at the bottom
        idealMaxAngleDeg = 160.0,  // back leg extended standing
        idealEccentricMs = 1500L..3000L,
        idealConcentricMs = 500L..2000L,
        eccentricLabel = "descent",
        concentricLabel = "stand"
    )
    private val tracker = RepPhaseTracker(config)
    private val evaluator = FormEvaluator(config)

    private val RIGHT_FOOT = 32
    private val LEFT_FOOT = 31

    // Knee indices for monitoring the BACK leg
    private val LEFT_HIP = 23
    private val LEFT_KNEE = 25
    private val LEFT_ANKLE = 27

    private val RIGHT_HIP = 24
    private val RIGHT_KNEE = 26
    private val RIGHT_ANKLE = 28

    private val RIGHT_SHOULDER = 12

    override fun isInStartPosition(landmarks: List<NormalizedLandmark>): Boolean {
        if (landmarks.size <= RIGHT_FOOT) return false
        val shoulder = landmarks[RIGHT_SHOULDER]
        val hip = landmarks[RIGHT_HIP]

        // Standing upright with both legs extended.
        val vertical = kotlin.math.abs(shoulder.y() - hip.y()) >
            kotlin.math.abs(shoulder.x() - hip.x())

        val leftKnee = AngleCalculator.calculateAngle(
            landmarks[LEFT_HIP].x(), landmarks[LEFT_HIP].y(),
            landmarks[LEFT_KNEE].x(), landmarks[LEFT_KNEE].y(),
            landmarks[LEFT_ANKLE].x(), landmarks[LEFT_ANKLE].y()
        )
        val rightKnee = AngleCalculator.calculateAngle(
            landmarks[RIGHT_HIP].x(), landmarks[RIGHT_HIP].y(),
            landmarks[RIGHT_KNEE].x(), landmarks[RIGHT_KNEE].y(),
            landmarks[RIGHT_ANKLE].x(), landmarks[RIGHT_ANKLE].y()
        )
        return vertical && leftKnee > 150.0 && rightKnee > 150.0
    }

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (landmarks.size <= RIGHT_FOOT) return Triple(repetitions, state, feedback)

        // 1. Detect Forward Leg (Mirrored assumption: smaller X is forward)
        val currentForwardLeg = if (landmarks[RIGHT_FOOT].x() < landmarks[LEFT_FOOT].x()) "RIGHT" else "LEFT"

        // 2. Monitor BACK knee angle
        val backKneeAngle = if (currentForwardLeg == "RIGHT") {
            AngleCalculator.calculateAngle(
                landmarks[LEFT_HIP].x(), landmarks[LEFT_HIP].y(),
                landmarks[LEFT_KNEE].x(), landmarks[LEFT_KNEE].y(),
                landmarks[LEFT_ANKLE].x(), landmarks[LEFT_ANKLE].y()
            )
        } else {
            AngleCalculator.calculateAngle(
                landmarks[RIGHT_HIP].x(), landmarks[RIGHT_HIP].y(),
                landmarks[RIGHT_KNEE].x(), landmarks[RIGHT_KNEE].y(),
                landmarks[RIGHT_ANKLE].x(), landmarks[RIGHT_ANKLE].y()
            )
        }

        // 3. Phase + timing tracking on the back-knee angle.
        val cycle = tracker.update(backKneeAngle)
        state = if (tracker.phase == RepPhaseTracker.Phase.DESCENDING) "DOWN" else "UP"

        if (cycle != null) {
            // Only count the rep when the user has switched the leading leg.
            if (currentForwardLeg != lastForwardLeg) {
                repetitions++
                lastForwardLeg = currentForwardLeg
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
                feedback = "Switch legs to count the rep"
            }
        } else {
            feedback = when (tracker.phase) {
                RepPhaseTracker.Phase.DESCENDING -> "Lower the back knee..."
                RepPhaseTracker.Phase.ASCENDING -> "Stand up!"
                RepPhaseTracker.Phase.AT_TOP ->
                    if (repetitions == 0) "Start Lunge (Alternate Legs)" else "Switch and repeat!"
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
        lastForwardLeg = ""
    }
}
