package com.example.fitnesstrackerapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var exerciseName = ""
    private var repetitions = 0
    private var feedback = ""
    private var exerciseColor = Color.GREEN

    // Normalized landmarks (x, y between 0.0 and 1.0)
    private var landmarks: List<Pair<Float, Float>> = emptyList()

    private val POSE_CONNECTIONS = listOf(
        Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 7),
        Pair(0, 4), Pair(4, 5), Pair(5, 6), Pair(6, 8),
        Pair(9, 10), Pair(11, 12), Pair(11, 13), Pair(13, 15),
        Pair(15, 17), Pair(15, 19), Pair(15, 21), Pair(17, 19),
        Pair(12, 14), Pair(14, 16), Pair(16, 18), Pair(16, 20),
        Pair(16, 22), Pair(18, 20), Pair(11, 23), Pair(12, 24),
        Pair(23, 24), Pair(23, 25), Pair(24, 26), Pair(25, 27),
        Pair(26, 28), Pair(27, 29), Pair(28, 30), Pair(29, 31),
        Pair(30, 32), Pair(27, 31), Pair(28, 32)
    )

    private val linePaint = Paint().apply {
        color = Color.parseColor("#00FF00") // Verde claro
        strokeWidth = 8f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val pointPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 50f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    private val countPaint = Paint().apply {
        color = Color.GREEN
        textSize = 70f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
        textAlign = Paint.Align.CENTER
    }

    private val feedbackPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 55f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
        textAlign = Paint.Align.CENTER
    }

    fun updateResults(
        landmarks: List<Pair<Float, Float>>,
        exerciseName: String,
        repetitions: Int,
        feedback: String,
        color: Int
    ) {
        this.landmarks = landmarks
        this.exerciseName = exerciseName
        this.repetitions = repetitions
        this.feedback = feedback
        this.exerciseColor = color
        linePaint.color = color
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        // Draw structural lines / connections
        POSE_CONNECTIONS.forEach { connection ->
            if (connection.first < landmarks.size && connection.second < landmarks.size) {
                val startP = landmarks[connection.first]
                val endP = landmarks[connection.second]
                canvas.drawLine(
                    startP.first * w, startP.second * h,
                    endP.first * w, endP.second * h,
                    linePaint
                )
            }
        }

        // Draw points
        landmarks.forEach { point ->
            val cx = point.first * w
            val cy = point.second * h
            canvas.drawCircle(cx, cy, 8f, pointPaint)
        }
        
        // Draw Exercise Info
        canvas.drawText("$exerciseName: $repetitions", w / 2, 120f, countPaint)
        canvas.drawText(feedback, w / 2, 200f, feedbackPaint)
    }
}
