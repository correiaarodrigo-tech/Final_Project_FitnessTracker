package com.example.fitnesstrackerapp.logic

import kotlin.math.abs

/**
 * Auto-start trigger: the session begins when the user is in the exercise's
 * start position AND holds still for [requiredMs].
 *
 * Stability is measured as the average per-frame movement of the torso/leg
 * landmarks (hands tremble too much to be useful). Thresholds are in
 * normalized image coordinates and may need tuning on a real device.
 */
class StartPositionDetector(
    private val requiredMs: Long = 1500L,
    private val movementThreshold: Float = 0.010f
) {
    // Torso + legs: stable, high-confidence landmarks.
    private val keyPoints = listOf(11, 12, 23, 24, 25, 26, 27, 28)

    private var holdStartMs = 0L
    private var lastPoints: List<Pair<Float, Float>>? = null

    /**
     * Feeds one frame. Returns hold progress in 0..1 — when it reaches 1 the
     * caller should start the countdown.
     */
    fun update(inPosition: Boolean, points: List<Pair<Float, Float>>, nowMs: Long): Float {
        val prev = lastPoints
        lastPoints = points

        if (!inPosition) {
            holdStartMs = 0L
            return 0f
        }

        val stable = prev != null && prev.size == points.size && averageMovement(prev, points) < movementThreshold
        if (!stable) {
            holdStartMs = 0L
            return 0f
        }

        if (holdStartMs == 0L) holdStartMs = nowMs
        return ((nowMs - holdStartMs).toFloat() / requiredMs).coerceIn(0f, 1f)
    }

    fun reset() {
        holdStartMs = 0L
        lastPoints = null
    }

    private fun averageMovement(a: List<Pair<Float, Float>>, b: List<Pair<Float, Float>>): Float {
        var sum = 0f
        var count = 0
        keyPoints.forEach { idx ->
            if (idx < a.size && idx < b.size) {
                sum += abs(a[idx].first - b[idx].first) + abs(a[idx].second - b[idx].second)
                count++
            }
        }
        return if (count == 0) Float.MAX_VALUE else sum / count
    }
}
