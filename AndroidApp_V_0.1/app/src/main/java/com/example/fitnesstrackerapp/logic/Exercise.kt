package com.example.fitnesstrackerapp.logic

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

interface Exercise {
    val name: String
    val color: Int
    var repetitions: Int
    var state: String
    var feedback: String

    /**
     * Processes the landmarks and returns the current state:
     * Triple(repetitions, state, feedback)
     */
    fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String>
    
    fun reset()
}
