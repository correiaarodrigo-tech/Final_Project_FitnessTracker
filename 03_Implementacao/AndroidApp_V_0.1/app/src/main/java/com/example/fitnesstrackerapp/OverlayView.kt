package com.example.fitnesstrackerapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.example.fitnesstrackerapp.logic.RepMetrics

/**
 * Camera overlay that renders the live skeleton plus a phase-aware HUD:
 *  - START_POSITION: guide the user into the start pose; a "hold still" bar
 *                    fills while they stay in position
 *  - COUNTDOWN:      large 5..1 / GO! counter
 *  - EXERCISE:       rep/time progress, current cue, and live form evaluation panel
 *
 * Styling mirrors the app's dark theme (DarkSurface panels, neon accents).
 */
class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    enum class Phase { START_POSITION, COUNTDOWN, EXERCISE }

    // ---- Theme colors (mirror ui/theme/Theme.kt) ----
    private val cPanel = Color.argb(205, 0x17, 0x20, 0x2A)   // DarkSurface, translucent
    private val cDim = Color.argb(150, 0x0C, 0x0F, 0x14)     // DarkBackground veil
    private val cTextSecondary = Color.parseColor("#9EADB8")
    private val cGreen = Color.parseColor("#39FF14")
    private val cYellow = Color.parseColor("#FFD23F")
    private val cRed = Color.parseColor("#FF4D4D")

    private var phase = Phase.START_POSITION
    private var accent = Color.parseColor("#00F0FF")

    private var landmarks: List<Pair<Float, Float>> = emptyList()

    // Exercise phase state
    private var exerciseName = ""
    private var feedback = ""
    private var progress = 0
    private var target = 0
    private var isTimeBased = false
    private var lastRepScore = -1
    private var lastRepMetrics: RepMetrics? = null

    // Start-position / countdown state
    private var facingInstruction = ""
    private var positionHint = ""
    private var inPosition = false
    private var holdProgress = 0f
    private var countdownText = ""

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
        strokeWidth = 9f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    private val jointPaint = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val jointCorePaint = Paint().apply {
        color = Color.WHITE; style = Paint.Style.FILL; isAntiAlias = true
    }

    private val panelPaint = Paint().apply { color = cPanel; style = Paint.Style.FILL; isAntiAlias = true }
    private val panelBorderPaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 3f; isAntiAlias = true
    }
    private val dimPaint = Paint().apply { color = cDim; style = Paint.Style.FILL }

    private val titlePaint = Paint().apply {
        color = Color.WHITE; textSize = 46f; isAntiAlias = true
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val bigCountPaint = Paint().apply {
        color = Color.WHITE; textSize = 64f; isAntiAlias = true
        textAlign = Paint.Align.RIGHT
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val progressTrackPaint = Paint().apply {
        color = Color.argb(120, 0x2C, 0x3A, 0x47); style = Paint.Style.FILL; isAntiAlias = true
    }
    private val progressFillPaint = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }

    private val pillPaint = Paint().apply { color = cPanel; style = Paint.Style.FILL; isAntiAlias = true }
    private val pillTextPaint = Paint().apply {
        color = Color.WHITE; textSize = 42f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }

    private val scorePaint = Paint().apply {
        textSize = 50f; isAntiAlias = true; typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val metricPaint = Paint().apply { color = Color.WHITE; textSize = 34f; isAntiAlias = true }
    private val cuePaint = Paint().apply { color = cYellow; textSize = 31f; isAntiAlias = true }

    private val centerBigPaint = Paint().apply {
        color = Color.WHITE; textSize = 230f; isAntiAlias = true; textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val centerTitlePaint = Paint().apply {
        color = Color.WHITE; textSize = 58f; isAntiAlias = true; textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val centerSubPaint = Paint().apply {
        color = cTextSecondary; textSize = 40f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }
    private val audioAdvisedPaint = Paint().apply {
        color = cTextSecondary; textSize = 28f; isAntiAlias = true; textAlign = Paint.Align.CENTER
    }

    // ---- Public render API ----

    fun showStartPosition(
        landmarks: List<Pair<Float, Float>>,
        exerciseName: String,
        facingInstruction: String,
        positionHint: String,
        inPosition: Boolean,
        holdProgress: Float,
        accentColor: Int
    ) {
        phase = Phase.START_POSITION
        this.landmarks = landmarks
        this.exerciseName = exerciseName
        this.facingInstruction = facingInstruction
        this.positionHint = positionHint
        this.inPosition = inPosition
        this.holdProgress = holdProgress
        this.accent = accentColor
        invalidate()
    }

    fun showCountdown(
        landmarks: List<Pair<Float, Float>>,
        text: String,
        accentColor: Int
    ) {
        phase = Phase.COUNTDOWN
        this.landmarks = landmarks
        this.countdownText = text
        this.accent = accentColor
        invalidate()
    }

    fun showExercise(
        landmarks: List<Pair<Float, Float>>,
        exerciseName: String,
        progress: Int,
        target: Int,
        isTimeBased: Boolean,
        feedback: String,
        accentColor: Int,
        lastRepScore: Int,
        lastRepMetrics: RepMetrics?
    ) {
        phase = Phase.EXERCISE
        this.landmarks = landmarks
        this.exerciseName = exerciseName
        this.progress = progress
        this.target = target
        this.isTimeBased = isTimeBased
        this.feedback = feedback
        this.accent = accentColor
        this.lastRepScore = lastRepScore
        this.lastRepMetrics = lastRepMetrics
        invalidate()
    }

    /** Backward-compatible entry point used by MainActivity / DemoPushUpActivity. */
    fun updateResults(
        landmarks: List<Pair<Float, Float>>,
        exerciseName: String,
        repetitions: Int,
        feedback: String,
        color: Int,
        lastRepScore: Int = -1,
        lastRepMetrics: RepMetrics? = null
    ) {
        showExercise(landmarks, exerciseName, repetitions, 0, false, feedback, color, lastRepScore, lastRepMetrics)
    }

    private fun scoreColor(score: Int): Int = when {
        score >= 80 -> cGreen
        score >= 60 -> cYellow
        else -> cRed
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        drawSkeleton(canvas, w, h)

        when (phase) {
            Phase.START_POSITION -> drawStartPosition(canvas, w, h)
            Phase.COUNTDOWN -> drawCountdown(canvas, w, h)
            Phase.EXERCISE -> drawExercise(canvas, w, h)
        }
    }

    private fun drawSkeleton(canvas: Canvas, w: Float, h: Float) {
        if (landmarks.isEmpty()) return
        linePaint.color = accent
        jointPaint.color = accent

        POSE_CONNECTIONS.forEach { (a, b) ->
            if (a < landmarks.size && b < landmarks.size) {
                val p1 = landmarks[a]
                val p2 = landmarks[b]
                canvas.drawLine(p1.first * w, p1.second * h, p2.first * w, p2.second * h, linePaint)
            }
        }
        landmarks.forEach { p ->
            val cx = p.first * w
            val cy = p.second * h
            canvas.drawCircle(cx, cy, 10f, jointPaint)
            canvas.drawCircle(cx, cy, 4f, jointCorePaint)
        }
    }

    private fun roundedPanel(canvas: Canvas, rect: RectF, borderColor: Int, radius: Float = 28f) {
        canvas.drawRoundRect(rect, radius, radius, panelPaint)
        panelBorderPaint.color = borderColor
        canvas.drawRoundRect(rect, radius, radius, panelBorderPaint)
    }

    // ---- EXERCISE phase ----
    private fun drawExercise(canvas: Canvas, w: Float, h: Float) {
        // Top HUD: name + progress count + progress bar
        val margin = 30f
        val top = 40f
        val panelH = 150f
        val rect = RectF(margin, top, w - margin, top + panelH)
        roundedPanel(canvas, rect, accent)

        canvas.drawText(exerciseName, margin + 36f, top + 62f, titlePaint)

        val unit = if (isTimeBased) "s" else "reps"
        if (target > 0) {
            bigCountPaint.color = Color.WHITE
            canvas.drawText("$progress / $target", w - margin - 36f, top + 70f, bigCountPaint)
            metricPaint.color = cTextSecondary
            metricPaint.textAlign = Paint.Align.RIGHT
            canvas.drawText(unit, w - margin - 36f, top + 110f, metricPaint)
            metricPaint.textAlign = Paint.Align.LEFT
            metricPaint.color = Color.WHITE

            // Progress bar
            val barL = margin + 36f
            val barR = w - margin - 36f
            val barTop = top + panelH - 34f
            val barBot = barTop + 14f
            canvas.drawRoundRect(RectF(barL, barTop, barR, barBot), 7f, 7f, progressTrackPaint)
            val frac = (progress.toFloat() / target).coerceIn(0f, 1f)
            progressFillPaint.color = accent
            canvas.drawRoundRect(RectF(barL, barTop, barL + (barR - barL) * frac, barBot), 7f, 7f, progressFillPaint)
        } else {
            bigCountPaint.color = Color.WHITE
            canvas.drawText("$progress", w - margin - 36f, top + 80f, bigCountPaint)
        }

        // Current cue pill (just under HUD)
        if (feedback.isNotEmpty()) {
            val pillW = (pillTextPaint.measureText(feedback) + 70f).coerceAtMost(w - 2 * margin)
            val pillTop = top + panelH + 22f
            val pillRect = RectF((w - pillW) / 2f, pillTop, (w + pillW) / 2f, pillTop + 70f)
            canvas.drawRoundRect(pillRect, 35f, 35f, pillPaint)
            canvas.drawText(feedback, w / 2f, pillTop + 47f, pillTextPaint)
        }

        // Form evaluation panel (bottom-left)
        if (lastRepScore >= 0) {
            val m = lastRepMetrics
            val lines = mutableListOf<String>()
            m?.let {
                lines.add("Eccentric  ${it.eccentricDurationMs} ms")
                lines.add("Concentric ${it.concentricDurationMs} ms")
                lines.add("Depth  ${it.minAngleDeg.toInt()}°")
                it.feedback.take(2).forEach { c -> lines.add("• $c") }
            }
            val panelLeft = 30f
            val panelW = 560f
            val lineH = 44f
            val panelTop = h - 110f - lines.size * lineH
            val panelRect = RectF(panelLeft, panelTop, panelLeft + panelW, h - 40f)
            roundedPanel(canvas, panelRect, scoreColor(lastRepScore))

            scorePaint.color = scoreColor(lastRepScore)
            canvas.drawText("Form  $lastRepScore/100", panelLeft + 28f, panelTop + 56f, scorePaint)

            var y = panelTop + 56f + 50f
            lines.forEach { line ->
                val paint = if (line.startsWith("•")) cuePaint else metricPaint
                canvas.drawText(line, panelLeft + 28f, y, paint)
                y += lineH
            }
        }
    }

    // ---- START_POSITION phase ----
    private fun drawStartPosition(canvas: Canvas, w: Float, h: Float) {
        // Keep the veil light so the user can see themselves while positioning.
        canvas.drawRect(0f, 0f, w, h, dimPaint)

        val cardW = (w * 0.86f)
        val cardH = 440f
        val left = (w - cardW) / 2f
        val topY = h * 0.12f // top area: the body usually occupies the center
        val rect = RectF(left, topY, left + cardW, topY + cardH)
        val borderColor = if (inPosition) cGreen else accent
        roundedPanel(canvas, rect, borderColor, radius = 36f)

        val cx = w / 2f
        canvas.drawText(exerciseName, cx, topY + 86f, centerTitlePaint)

        centerSubPaint.color = accent
        canvas.drawText(facingInstruction, cx, topY + 152f, centerSubPaint)
        centerSubPaint.color = cTextSecondary
        canvas.drawText(positionHint, cx, topY + 208f, centerSubPaint)

        // Status line
        centerTitlePaint.textSize = 44f
        centerTitlePaint.color = if (inPosition) cGreen else Color.WHITE
        val status = when {
            holdProgress > 0f -> "Hold still..."
            inPosition -> "Almost there..."
            else -> "Get into position"
        }
        canvas.drawText(status, cx, topY + 296f, centerTitlePaint)
        centerTitlePaint.textSize = 58f
        centerTitlePaint.color = Color.WHITE

        // Hold-still progress bar
        val barL = left + 60f
        val barR = left + cardW - 60f
        val barTop = topY + 350f
        val barBot = barTop + 22f
        canvas.drawRoundRect(RectF(barL, barTop, barR, barBot), 11f, 11f, progressTrackPaint)
        if (holdProgress > 0f) {
            progressFillPaint.color = cGreen
            canvas.drawRoundRect(
                RectF(barL, barTop, barL + (barR - barL) * holdProgress.coerceIn(0f, 1f), barBot),
                11f, 11f, progressFillPaint
            )
        }
        canvas.drawText("Note: Audio guidance will be used and is advised (optional)", cx, topY + cardH + 54f, audioAdvisedPaint)
    }

    // ---- COUNTDOWN phase ----
    private fun drawCountdown(canvas: Canvas, w: Float, h: Float) {
        canvas.drawRect(0f, 0f, w, h, dimPaint)
        val cx = w / 2f
        val cy = h / 2f
        val isGo = countdownText.equals("GO!", ignoreCase = true)
        centerBigPaint.color = if (isGo) cGreen else accent
        // vertically center the text baseline
        val baseline = cy - (centerBigPaint.descent() + centerBigPaint.ascent()) / 2f
        canvas.drawText(countdownText, cx, baseline, centerBigPaint)

        if (!isGo) {
            canvas.drawText("Get ready", cx, cy + 200f, centerSubPaint)
        }
    }
}
