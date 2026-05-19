package com.example.fitnesstrackerapp

import android.Manifest
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
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.example.fitnesstrackerapp.logic.Exercise
import com.example.fitnesstrackerapp.logic.WorkoutManager
import com.example.fitnesstrackerapp.logic.TrainingStep
import com.example.fitnesstrackerapp.logic.impl.*

class DemoPushUpActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView
    
    private var poseLandmarker: PoseLandmarker? = null
    private lateinit var cameraExecutor: ExecutorService

    // Scalable Workout Manager
    private val workoutManager = WorkoutManager(listOf(
        TrainingStep(PushUpExercise(), targetReps = 100)
    ))

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupPoseLandmarker()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupPoseLandmarker() {
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
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, { image ->
                        processImage(image)
                    })
                }

            // Usar a cÃ¢mara frontal por norma em apps de fitness tracker
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        // O CameraX ImageProxy.toBitmap() simplifica isto
        imageProxy.use { proxy ->
            val bitmap = proxy.toBitmap()
            
            // Rodar bitmap em imagens de cÃ¢mara frontal (fazer mirror)
            val matrix = Matrix().apply {
                postRotate(proxy.imageInfo.rotationDegrees.toFloat())
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
            val rotatedBitmap = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            
            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            val timestampMs = System.currentTimeMillis() // ou proxy.imageInfo.timestamp / 1_000_000
            
            poseLandmarker?.detectAsync(mpImage, timestampMs)
        }
    }

    private fun returnLivestreamResult(result: PoseLandmarkerResult, input: MPImage) {
        val landmarks = result.landmarks()
        if (landmarks.isEmpty()) return

        // Assumindo a primeira pessoa encontrada
        val poseLandmarks = landmarks[0]

        // Process logic with current workout step
        val (reps, state, feedback) = workoutManager.processLandmarks(poseLandmarks)
        val currentStep = workoutManager.currentStep

        // Convert to draw on screen
        val points = poseLandmarks.map { Pair(it.x(), it.y()) }

        runOnUiThread {
            overlayView.updateResults(
                points,
                currentStep?.exercise?.name ?: "Finished",
                reps,
                feedback,
                currentStep?.exercise?.color ?: android.graphics.Color.GREEN
            )
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
        poseLandmarker?.close()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "FitnessTracker"
    }
}
