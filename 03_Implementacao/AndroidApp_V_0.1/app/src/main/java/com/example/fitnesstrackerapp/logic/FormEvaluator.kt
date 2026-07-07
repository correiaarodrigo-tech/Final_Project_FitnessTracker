package com.example.fitnesstrackerapp.logic

/**
 * Scores a completed rep and gives correction cues.
 *
 * Score starts at 100:
 *  - Depth ............ up to -40
 *  - Eccentric tempo ... up to -30
 *  - Concentric tempo .. up to -25
 */
class FormEvaluator(private val config: ExerciseConfig) {

    companion object {
        private const val MAX_DEPTH_PENALTY = 40.0
    }

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

        // Depth: penalty grows the farther past the ideal the user stopped.
        val depthError = (minAngle - config.idealMinAngleDeg).coerceAtLeast(0.0)
        val depthRange = (config.countThresholdDeg - config.idealMinAngleDeg).coerceAtLeast(1.0)
        val depthPenalty = ((depthError / depthRange) * MAX_DEPTH_PENALTY).coerceIn(0.0, MAX_DEPTH_PENALTY)
        if (depthPenalty > 0.0) {
            score -= depthPenalty.toInt()
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
