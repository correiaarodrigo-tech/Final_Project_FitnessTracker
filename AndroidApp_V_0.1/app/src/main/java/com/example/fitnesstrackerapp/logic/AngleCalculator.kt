package com.example.fitnesstrackerapp.logic

import kotlin.math.acos
import kotlin.math.sqrt

object AngleCalculator {
    /**
     * Calculates the angle ABC (in degrees) given three points.
     * b is the vertex.
     */
    fun calculateAngle(
        ax: Float, ay: Float,
        bx: Float, by: Float,
        cx: Float, cy: Float
    ): Double {
        val baX = ax - bx
        val baY = ay - by
        val bcX = cx - bx
        val bcY = cy - by

        val dotProduct = baX * bcX + baY * bcY
        val magnitudeBa = sqrt(baX * baX + baY * baY)
        val magnitudeBc = sqrt(bcX * bcX + bcY * bcY)

        if (magnitudeBa == 0f || magnitudeBc == 0f) return 0.0

        val cosTheta = dotProduct / (magnitudeBa * magnitudeBc)
        // Clamp cosTheta to [-1, 1] to avoid NaN due to floating point errors
        val clampedCosTheta = cosTheta.coerceIn(-1f, 1f)
        
        val angleRad = acos(clampedCosTheta)
        return Math.toDegrees(angleRad.toDouble())
    }
}
