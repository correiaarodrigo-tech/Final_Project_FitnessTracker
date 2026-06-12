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

    /**
     * Per-rep evaluation history. Empty by default for exercises that do not
     * score form (e.g. Plank, Rest); rep-based exercises override this with a
     * real backing list.
     */
    val repHistory: List<RepMetrics>
        get() = emptyList()

    /** Metrics for the most recently completed rep, or null if none yet. */
    val lastRepMetrics: RepMetrics?
        get() = repHistory.lastOrNull()

    /** Form score (0-100) of the last rep, or -1 if not available yet. */
    val lastRepScore: Int
        get() = lastRepMetrics?.formScore ?: -1
}
