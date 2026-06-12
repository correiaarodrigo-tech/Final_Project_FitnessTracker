package com.example.fitnesstrackerapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.example.fitnesstrackerapp.logic.AngleCalculator
import com.example.fitnesstrackerapp.logic.RepMetrics

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var exerciseName = ""
    private var repetitions = 0
    private var feedback = ""
    private var exerciseColor = Color.GREEN
    private var lastRepScore = -1
    private var lastRepMetrics: RepMetrics? = null

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

    private val instructionPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
        textAlign = Paint.Align.CENTER
    }

    // ---- Form evaluation panel paints (top-left, left aligned) ----
    private val scorePaint = Paint().apply {
        color = Color.GREEN
        textSize = 55f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    private val metricPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    private val cuePaint = Paint().apply {
        color = Color.YELLOW
        textSize = 34f
        isAntiAlias = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun updateResults(
        landmarks: List<Pair<Float, Float>>,
        exerciseName: String,
        repetitions: Int,
        feedback: String,
        color: Int,
        lastRepScore: Int = -1,
        lastRepMetrics: RepMetrics? = null
    ) {
        this.landmarks = landmarks
        this.exerciseName = exerciseName
        this.repetitions = repetitions
        this.feedback = feedback
        this.exerciseColor = color
        this.lastRepScore = lastRepScore
        this.lastRepMetrics = lastRepMetrics
        linePaint.color = color
        invalidate()
    }

    private fun scoreColor(score: Int): Int = when {
        score >= 80 -> Color.GREEN
        score >= 60 -> Color.YELLOW
        else -> Color.RED
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

        // Draw Form Evaluation panel (top-left) once at least one rep is scored
        if (lastRepScore >= 0) {
            val panelX = 40f
            var panelY = 300f
            scorePaint.color = scoreColor(lastRepScore)
            canvas.drawText("Form: $lastRepScore/100", panelX, panelY, scorePaint)

            lastRepMetrics?.let { m ->
                panelY += 55f
                canvas.drawText("Eccentric: ${m.eccentricDurationMs} ms", panelX, panelY, metricPaint)
                panelY += 45f
                canvas.drawText("Concentric: ${m.concentricDurationMs} ms", panelX, panelY, metricPaint)
                panelY += 45f
                canvas.drawText("Depth: ${m.minAngleDeg.toInt()}°", panelX, panelY, metricPaint)

                m.feedback.take(2).forEach { line ->
                    panelY += 42f
                    canvas.drawText("• $line", panelX, panelY, cuePaint)
                }
            }
        }

        // Draw Demo Push-Up Additions
        if (exerciseName == "Push-Up") {
            // Warning message to stay in frame
            canvas.drawText("Pls stay in frame facing the best position for the exercise", w / 2, h - 100f, instructionPaint)
            
            // Highlight arms and draw angle
            if (landmarks.size > 16) {
                // Shoulders: 11, 12, Elbows: 13, 14, Wrists: 15, 16
                val highlightPaint = Paint().apply {
                    color = Color.CYAN
                    strokeWidth = 12f
                    style = Paint.Style.STROKE
                    isAntiAlias = true
                }
                val highlightPointPaint = Paint().apply {
                    color = Color.YELLOW
                    style = Paint.Style.FILL
                }

                // Draw lines for right arm (12 -> 14 -> 16)
                val p12 = landmarks[12]
                val p14 = landmarks[14]
                val p16 = landmarks[16]
                
                canvas.drawLine(p12.first * w, p12.second * h, p14.first * w, p14.second * h, highlightPaint)
                canvas.drawLine(p14.first * w, p14.second * h, p16.first * w, p16.second * h, highlightPaint)

                // Draw lines for left arm (11 -> 13 -> 15)
                val p11 = landmarks[11]
                val p13 = landmarks[13]
                val p15 = landmarks[15]
                
                canvas.drawLine(p11.first * w, p11.second * h, p13.first * w, p13.second * h, highlightPaint)
                canvas.drawLine(p13.first * w, p13.second * h, p15.first * w, p15.second * h, highlightPaint)

                // Draw highlighted points (arms only)
                val armsPoints = listOf(11, 12, 13, 14, 15, 16)
                armsPoints.forEach { idx ->
                    val point = landmarks[idx]
                    canvas.drawCircle(point.first * w, point.second * h, 12f, highlightPointPaint)
                }

                // Calculate and draw angle for right elbow
                val angle = AngleCalculator.calculateAngle(
                    p12.first, p12.second,
                    p14.first, p14.second,
                    p16.first, p16.second
                )
                
                val anglePaint = Paint().apply {
                    color = Color.WHITE
                    textSize = 45f
                    isAntiAlias = true
                    setShadowLayer(5f, 0f, 0f, Color.BLACK)
                }
                canvas.drawText("${angle.toInt()}°", p14.first * w + 30f, p14.second * h, anglePaint)
            }
        }
    }
}
