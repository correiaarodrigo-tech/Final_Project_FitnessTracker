package com.example.fitnesstrackerapp.logic

/**
 * Metrics captured for a single completed repetition.
 *
 * @param repNumber             1-based index of the rep within the current set.
 * @param eccentricDurationMs   Time spent in the eccentric (lowering) phase, in ms.
 * @param concentricDurationMs  Time spent in the concentric (lifting) phase, in ms.
 * @param minAngleDeg           Deepest joint angle reached (bottom of the movement).
 * @param maxAngleDeg           Most extended joint angle reached (top of the movement).
 * @param formScore             Overall quality score from 0 to 100.
 * @param feedback              Specific, human-readable correction cues for this rep.
 */
data class RepMetrics(
    val repNumber: Int,
    val eccentricDurationMs: Long,
    val concentricDurationMs: Long,
    val minAngleDeg: Double,
    val maxAngleDeg: Double,
    val formScore: Int,
    val feedback: List<String>
)
