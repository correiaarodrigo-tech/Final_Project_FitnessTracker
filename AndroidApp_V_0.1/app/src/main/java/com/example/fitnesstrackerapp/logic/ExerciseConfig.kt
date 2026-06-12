package com.example.fitnesstrackerapp.logic

/**
 * Reference values an exercise is evaluated against.
 *
 * Angles are expressed for the tracked joint (e.g. elbow for push-ups,
 * knee for squats). [idealMinAngleDeg] is the target depth at the bottom of
 * the movement and [idealMaxAngleDeg] the target lock-out at the top.
 «
 * The tempo ranges describe the recommended duration of each phase. A rep that
 * is faster tadb deviceshan the lower bound is penalised (using momentum / not controlled);
 * slower than the upper bound is mildly penalised.
 */
data class ExerciseConfig(
    val name: String,
    val idealMinAngleDeg: Double,
    val idealMaxAngleDeg: Double,
    val minAngleTolerance: Double = 15.0,
    val idealEccentricMs: LongRange = 1500L..3000L,
    val idealConcentricMs: LongRange = 500L..2000L,
    val eccentricLabel: String = "descent",
    val concentricLabel: String = "lift"
)
