package com.example.fitnesstrackerapp.logic

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

class WorkoutManager(private val steps: List<TrainingStep>) {
    private var currentStepIndex = 0
    val currentStep: TrainingStep?
        get() = if (currentStepIndex < steps.size) steps[currentStepIndex] else null

    fun processLandmarks(landmarks: List<NormalizedLandmark>): Triple<Int, String, String> {
        val step = currentStep ?: return Triple(0, "COMPLETE", "Workout Finished!")
        
        val result = step.exercise.processLandmarks(landmarks)
        val reps = result.first
        val state = result.second
        
        // Check if step is finished
        if (step.isRest) {
            if (state == "FINISHED") {
                nextStep()
            }
        } else {
            if (reps >= step.targetReps && step.targetReps > 0) {
                nextStep()
            } else if (state == "FINISHED") { // Some exercises might finish by state
                nextStep()
            }
        }
        
        return result
    }

    private fun nextStep() {
        currentStepIndex++
        currentStep?.exercise?.reset()
    }

    fun isFinished(): Boolean = currentStepIndex >= steps.size
}
