package com.example.fitnesstrackerapp.logic

/**
 * Scores a completed repetition against an [ExerciseConfig] reference and
 * produces specific correction cues.
 *
 * Score breakdown (starts at 100):
 *  - Range of motion / depth ....... up to -40
 *  - Eccentric (lowering) tempo .... up to -30
 *  - Concentric (lifting) tempo .... up to -25
 */
class FormEvaluator(private val config: ExerciseConfig) {

    fun evaluate(cycle: RepPhaseTracker.RepCycle): Pair<Int, List<String>> =
        evaluate(
            cycle.minAngleDeg,
            cycle.maxAngleDeg,
            cycle.eccentricDurationMs,
            cycle.concentricDurationMs
        )

    fun evaluate(
        minAngle: Double,
        maxAngle: Double,
        eccentricMs: Long,
        concentricMs: Long
    ): Pair<Int, List<String>> {
        val feedback = mutableListOf<String>()
        var score = 100

        // 1. Range of motion: only penalise not reaching enough depth.
        val depthError = (minAngle - config.idealMinAngleDeg).coerceAtLeast(0.0)
        if (depthError > config.minAngleTolerance) {
            val penalty = ((depthError / config.minAngleTolerance) * 20.0)
                .toInt().coerceIn(0, 40)
            score -= penalty
            feedback.add("Desce mais!")
        }

        // 2. Eccentric tempo.
        when {
            eccentricMs < config.idealEccentricMs.first -> {
                score -= 30
                feedback.add("Mais lento a descer!")
            }
            eccentricMs > config.idealEccentricMs.last -> {
                score -= 10
                feedback.add("Desce mais rápido!")
            }
        }

        // 3. Concentric tempo.
        when {
            concentricMs < config.idealConcentricMs.first -> {
                score -= 15
                feedback.add("Sobe com controlo!")
            }
            concentricMs > config.idealConcentricMs.last -> {
                score -= 10
                feedback.add("Sobe mais rápido!")
            }
        }

        if (feedback.isEmpty()) feedback.add("Excelente!")
        return Pair(score.coerceIn(0, 100), feedback)
    }
}
