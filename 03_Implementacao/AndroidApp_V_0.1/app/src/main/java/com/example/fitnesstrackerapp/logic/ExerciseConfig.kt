package com.example.fitnesstrackerapp.logic

/**
 * Reference values an exercise is scored against.
 *
 * idealMinAngleDeg/idealMaxAngleDeg mark the bottom and top of a good rep.
 * countThresholdDeg is more lenient - crossing it still counts as a rep,
 * just with a lower score, instead of the rep being thrown away.
 */
data class ExerciseConfig(
    val name: String,
    val idealMinAngleDeg: Double,
    val idealMaxAngleDeg: Double,
    val countThresholdDeg: Double = idealMinAngleDeg + 20.0,
    val minAngleTolerance: Double = 15.0,
    val idealEccentricMs: LongRange = 1500L..3000L,
    val idealConcentricMs: LongRange = 500L..2000L,
    val eccentricLabel: String = "descent",
    val concentricLabel: String = "lift"
)
