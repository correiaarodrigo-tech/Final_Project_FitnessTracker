package com.example.fitnesstrackerapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.StartPositionDetector
import com.example.fitnesstrackerapp.logic.impl.*

/**
 * Single-exercise runner with a guided session flow:
 *
 *   WAITING  -> user gets into the start position and holds still ~1.5s
 *   COUNTDOWN -> 5..1 / GO!
 *   EXERCISING -> count [TARGET] reps (or hold [TARGET] seconds for the plank)
 *   FINISHED -> hand off to [ResultActivity] with the session summary
 */
class ExerciseTestActivity : AppCompatActivity() {

    private enum class SessionState { WAITING, COUNTDOWN, EXERCISING, FINISHED }

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView

    private var poseLandmarker: PoseLandmarker? = null
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var exercise: Exercise
    private val target = 20

    @Volatile private var sessionState = SessionState.WAITING
    private val startDetector = StartPositionDetector()
    private var countdownJob: Job? = null
    private var lastPoints: List<Pair<Float, Float>> = emptyList()

    private var ttsHelper: com.example.fitnesstrackerapp.logic.TTSHelper? = null
    private var lastReps = 0

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) startCamera()
            else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        val type = intent.getStringExtra(EXTRA_EXERCISE_TYPE) ?: ExerciseType.PUSHUP
        exercise = exerciseFor(type)
        ttsHelper = com.example.fitnesstrackerapp.logic.TTSHelper(this)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Initial prompt before any pose is detected.
        overlayView.showStartPosition(
            emptyList(), exercise.name, exercise.setupInstruction,
            exercise.startPositionHint, inPosition = false, holdProgress = 0f, exercise.color
        )

        cameraExecutor = Executors.newSingleThreadExecutor()
        setupPoseLandmarker()

        if (allPermissionsGranted()) startCamera()
        else requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun exerciseFor(type: String): Exercise = when (type) {
        ExerciseType.SQUAT -> SquatExercise()
        ExerciseType.LUNGE -> LungeExercise()
        ExerciseType.PLANK -> PlankExercise()
        else -> PushUpExercise()
    }

    private fun setupPoseLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener(this::returnLivestreamResult)
                .setErrorListener(this::returnLivestreamError)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(this, options)
        } catch (t: Throwable) {
            // MediaPipe ships native libs for ARM only. On an x86_64 emulator the
            // .so is missing and class init throws UnsatisfiedLinkError (an Error).
            Log.e(TAG, "Failed to init PoseLandmarker (unsupported ABI?)", t)
            Toast.makeText(
                this,
                "Pose tracking is not available on this device/emulator (ARM required). Run on a physical phone.",
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { image -> processImage(image) } }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        imageProxy.use { proxy ->
            val bitmap = proxy.toBitmap()
            val matrix = Matrix().apply {
                postRotate(proxy.imageInfo.rotationDegrees.toFloat())
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
            val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            poseLandmarker?.detectAsync(mpImage, System.currentTimeMillis())
        }
    }

    private fun returnLivestreamResult(result: PoseLandmarkerResult, input: MPImage) {
        val landmarks = result.landmarks()
        if (landmarks.isEmpty()) return
        val poseLandmarks = landmarks[0]
        val points = poseLandmarks.map { Pair(it.x(), it.y()) }
        lastPoints = points

        when (sessionState) {
            SessionState.WAITING -> {
                val inPosition = exercise.isInStartPosition(poseLandmarks)
                val holdProgress = startDetector.update(inPosition, points, System.currentTimeMillis())
                runOnUiThread {
                    overlayView.showStartPosition(
                        points, exercise.name, exercise.setupInstruction,
                        exercise.startPositionHint, inPosition, holdProgress, exercise.color
                    )
                }
                if (holdProgress >= 1f && sessionState == SessionState.WAITING) {
                    startCountdown()
                }
            }

            SessionState.COUNTDOWN -> {
                // Rendering is driven by the countdown coroutine; just keep the skeleton fresh.
                runOnUiThread { overlayView.showCountdown(points, countdownTextOrEmpty(), exercise.color) }
            }

            SessionState.EXERCISING -> {
                val (_, _, feedback) = exercise.processLandmarks(poseLandmarks)
                val prog = exercise.progress().coerceAtMost(target)

                // TTS Voice cues logic
                if (exercise.isTimeBased) {
                    val rawFeedback = feedback
                    if (!rawFeedback.startsWith("Remaining: ")) {
                        ttsHelper?.speak(rawFeedback, overrideCooldown = false)
                    }
                } else {
                    val reps = exercise.progress()
                    if (reps > lastReps) {
                        lastReps = reps
                        val lastMetrics = exercise.lastRepMetrics
                        val score = lastMetrics?.formScore
                        val primaryCue = lastMetrics?.feedback?.firstOrNull()
                        val speakText = if (score != null && primaryCue != null) {
                            "Rep $reps. Score $score. $primaryCue"
                        } else {
                            "Rep $reps."
                        }
                        ttsHelper?.speak(speakText, overrideCooldown = true)
                    } else {
                        val rawFeedback = feedback
                        if (!rawFeedback.startsWith("Rep ")) {
                            ttsHelper?.speak(rawFeedback, overrideCooldown = false)
                        }
                    }
                }

                runOnUiThread {
                    overlayView.showExercise(
                        points, exercise.name, prog, target, exercise.isTimeBased,
                        feedback, exercise.color, exercise.lastRepScore, exercise.lastRepMetrics
                    )
                }
                if (exercise.progress() >= target) finishSession()
            }

            SessionState.FINISHED -> { /* no-op */ }
        }
    }

    private var countdownText = ""
    private fun countdownTextOrEmpty() = countdownText

    private fun startCountdown() {
        sessionState = SessionState.COUNTDOWN
        countdownJob?.cancel()
        ttsHelper?.speak("Get ready!", overrideCooldown = true)
        countdownJob = lifecycleScope.launch {
            for (n in 5 downTo 1) {
                countdownText = n.toString()
                overlayView.showCountdown(lastPoints, countdownText, exercise.color)
                ttsHelper?.speak(countdownText, overrideCooldown = true)
                delay(1000)
            }
            countdownText = "GO!"
            overlayView.showCountdown(lastPoints, countdownText, exercise.color)
            ttsHelper?.speak("Go!", overrideCooldown = true)
            delay(700)
            exercise.reset() // fresh timers/state right before counting begins
            lastReps = 0
            sessionState = SessionState.EXERCISING
        }
    }

    private fun finishSession() {
        if (sessionState == SessionState.FINISHED) return
        sessionState = SessionState.FINISHED

        val history = exercise.repHistory
        val scores = history.map { it.formScore }
        val avgScore = if (scores.isNotEmpty()) scores.average().toInt()
            else if (exercise.isTimeBased) 100 else 0
        val avgEcc = if (history.isNotEmpty()) history.map { it.eccentricDurationMs }.average().toLong() else 0L
        val avgConc = if (history.isNotEmpty()) history.map { it.concentricDurationMs }.average().toLong() else 0L
        val best = scores.maxOrNull() ?: 0

        runOnUiThread {
            val intent = Intent(this, ResultActivity::class.java).apply {
                putExtra(ResultActivity.EXTRA_NAME, exercise.name)
                putExtra(ResultActivity.EXTRA_TIME_BASED, exercise.isTimeBased)
                putExtra(ResultActivity.EXTRA_ACHIEVED, exercise.progress().coerceAtMost(target))
                putExtra(ResultActivity.EXTRA_TARGET, target)
                putExtra(ResultActivity.EXTRA_AVG_SCORE, avgScore)
                putExtra(ResultActivity.EXTRA_AVG_ECC, avgEcc)
                putExtra(ResultActivity.EXTRA_AVG_CONC, avgConc)
                putExtra(ResultActivity.EXTRA_BEST, best)
                putExtra(ResultActivity.EXTRA_SCORES, scores.toIntArray())
            }
            startActivity(intent)
            finish()
        }
    }

    private fun returnLivestreamError(error: RuntimeException) {
        Log.e(TAG, "PoseLandmarker Error: ${error.message}", error)
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        countdownJob?.cancel()
        poseLandmarker?.close()
        cameraExecutor.shutdown()
        ttsHelper?.shutdown()
    }

    companion object {
        private const val TAG = "FitnessTracker"
        const val EXTRA_EXERCISE_TYPE = "EXERCISE_TYPE"
    }
}

/** Exercise type keys passed to [ExerciseTestActivity]. */
object ExerciseType {
    const val PUSHUP = "PUSHUP"
    const val SQUAT = "SQUAT"
    const val LUNGE = "LUNGE"
    const val PLANK = "PLANK"
}
