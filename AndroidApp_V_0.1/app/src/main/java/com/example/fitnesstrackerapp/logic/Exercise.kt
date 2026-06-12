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

    /** How the user should orient toward the camera (e.g. side profile vs front). */
    val setupInstruction: String
        get() = "Face the camera"

    /** Short description of the start pose (e.g. "Stand tall, legs straight"). */
    val startPositionHint: String
        get() = "Get into start position"

    /**
     * True when the body is in this exercise's start pose. Used by the
     * auto-start flow: being in position + holding still triggers the countdown.
     */
    fun isInStartPosition(landmarks: List<NormalizedLandmark>): Boolean = true

    /** True for holds measured in seconds (e.g. Plank); false for rep-based exercises. */
    val isTimeBased: Boolean
        get() = false

    /** Progress toward the goal: rep count, or seconds held for time-based holds. */
    fun progress(): Int = repetitions

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
