package com.example.fitnesstrackerapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var rightArmCount = 0
    private var leftArmCount = 0
    private var rightAngle = 0f
    private var leftAngle = 0f
    
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
        strokeWidth = 6f
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
        textSize = 55f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun updateResults(
        landmarks: List<Pair<Float, Float>>,
        rightAngle: Float,
        leftAngle: Float,
        rightCount: Int,
        leftCount: Int
    ) {
        this.landmarks = landmarks
        this.rightAngle = rightAngle
        this.leftAngle = leftAngle
        this.rightArmCount = rightCount
        this.leftArmCount = leftCount
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
        
        // Draw counts top-left (stacked to avoid center overlap on smaller screens)
        canvas.drawText("Braco Direito: $rightArmCount", 50f, 120f, countPaint)
        canvas.drawText("Braco Esquerdo: $leftArmCount", 50f, 190f, countPaint)
        
        // Draw angles near elbows (indices 13 = left elbow, 14 = right elbow in mediapipe)
        if (landmarks.size > 14) {
            val rightElbow = landmarks[14]
            val leftElbow = landmarks[13]
            
            val rx = rightElbow.first * w
            val ry = rightElbow.second * h
            canvas.drawText("R:${rightAngle.toInt()}", rx, ry, textPaint)
            
            val lx = leftElbow.first * w
            val ly = leftElbow.second * h
            canvas.drawText("L:${leftAngle.toInt()}", lx, ly, textPaint)
        }
    }
}
