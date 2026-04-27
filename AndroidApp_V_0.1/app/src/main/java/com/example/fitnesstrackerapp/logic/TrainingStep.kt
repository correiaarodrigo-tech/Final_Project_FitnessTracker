package com.example.fitnesstrackerapp.logic

data class TrainingStep(
    val exercise: Exercise,
    val targetReps: Int = 0,
    val targetSeconds: Int = 0,
    val isRest: Boolean = false
)
