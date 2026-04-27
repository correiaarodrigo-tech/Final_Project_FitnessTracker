package com.example.fitnesstrackerapp.logic.impl

import android.graphics.Color
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.Exercise
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class LungeExercise : Exercise {
    override val name: String = "Lunge"
    override val color: Int = Color.MAGENTA
    override var repetitions: Int = 0
    override var state: String = "UP"
    override var feedback: String = "Start Lunge (Alternate Legs)"

    private var isDown: Boolean = false
    private var lastForwardLeg: String = "" // "LEFT" or "RIGHT"

    private val RIGHT_FOOT = 32
    private val LEFT_FOOT = 31
    
    // Knee indices for monitoring the BACK leg
    private val LEFT_HIP = 23
    private val LEFT_KNEE = 25
    private val LEFT_ANKLE = 27
    
    private val RIGHT_HIP = 24
    private val RIGHT_KNEE = 26
    private val RIGHT_ANKLE = 28

    override fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        if (landmarks.size <= RIGHT_FOOT) return Triple(repetitions, state, feedback)

        // 1. Detect Forward Leg (Mirrored assumption: smaller X is forward)
        val currentForwardLeg = if (landmarks[RIGHT_FOOT].x() < landmarks[LEFT_FOOT].x()) "RIGHT" else "LEFT"
        
        // 2. Monitor BACK knee angle
        val backKneeAngle = if (currentForwardLeg == "RIGHT") {
            // Monitor LEFT knee (back leg)
            AngleCalculator.calculateAngle(
                landmarks[LEFT_HIP].x(), landmarks[LEFT_HIP].y(),
                landmarks[LEFT_KNEE].x(), landmarks[LEFT_KNEE].y(),
                landmarks[LEFT_ANKLE].x(), landmarks[LEFT_ANKLE].y()
            )
        } else {
            // Monitor RIGHT knee (back leg)
            AngleCalculator.calculateAngle(
                landmarks[RIGHT_HIP].x(), landmarks[RIGHT_HIP].y(),
                landmarks[RIGHT_KNEE].x(), landmarks[RIGHT_KNEE].y(),
                landmarks[RIGHT_ANKLE].x(), landmarks[RIGHT_ANKLE].y()
            )
        }

        // 3. State Machine
        if (backKneeAngle < 80 && !isDown) {
            isDown = true
            state = "DOWN"
            feedback = "Great! Now switch legs"
        } else if (backKneeAngle > 160 && isDown) {
            if (currentForwardLeg != lastForwardLeg) {
                repetitions++
                lastForwardLeg = currentForwardLeg
            }
            isDown = false
            state = "UP"
            feedback = "Switch and repeat!"
        }

        return Triple(repetitions, state, feedback)
    }

    override fun reset() {
        repetitions = 0
        state = "UP"
        feedback = "Reset"
        isDown = false
        lastForwardLeg = ""
    }
}
