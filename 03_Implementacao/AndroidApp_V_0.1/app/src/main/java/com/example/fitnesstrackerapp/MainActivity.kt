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
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.Timestamp
import java.util.Date

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: OverlayView
    
    private var poseLandmarker: PoseLandmarker? = null
    private lateinit var cameraExecutor: ExecutorService

    private var ttsHelper: com.example.fitnesstrackerapp.logic.TTSHelper? = null
    private var lastReps = 0
    private var lastExerciseName = ""

    private lateinit var workoutManager: WorkoutManager

    private var workoutStartTime = 0L

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
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // Parse plan steps from intent
        val planName = intent.getStringExtra("EXTRA_PLAN_NAME") ?: "Mini Plano"
        val stepsJson = intent.getStringExtra("EXTRA_PLAN_STEPS_JSON")
        workoutManager = WorkoutManager(parsePlanSteps(stepsJson))

        workoutStartTime = System.currentTimeMillis()
        ttsHelper = com.example.fitnesstrackerapp.logic.TTSHelper(this)

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

    private fun parsePlanSteps(jsonStr: String?): List<TrainingStep> {
        val list = mutableListOf<TrainingStep>()
        if (!jsonStr.isNullOrBlank()) {
            try {
                val arr = org.json.JSONArray(jsonStr)
                for (i in 0 until arr.length()) {
                    val obj = arr.getJSONObject(i)
                    val type = obj.getString("type")
                    val value = obj.getInt("value")
                    val step = when (type) {
                        "SQUAT" -> TrainingStep(SquatExercise(), targetReps = value)
                        "PUSHUP" -> TrainingStep(PushUpExercise(), targetReps = value)
                        "LUNGE" -> TrainingStep(LungeExercise(), targetReps = value)
                        else -> TrainingStep(RestExercise(value), isRest = true, targetSeconds = value)
                    }
                    list.add(step)
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Failed to parse custom plan JSON: ${e.localizedMessage}")
            }
        }
        if (list.isEmpty()) {
            list.add(TrainingStep(SquatExercise(), targetReps = 10))
            list.add(TrainingStep(RestExercise(15), isRest = true, targetSeconds = 15))
            list.add(TrainingStep(LungeExercise(), targetReps = 10))
        }
        return list
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

            // Usar a câmara frontal por norma em apps de fitness tracker
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
            
            // Rodar bitmap em imagens de câmara frontal (fazer mirror)
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
        
        if (workoutManager.isFinished()) {
            runOnUiThread {
                finishWorkout()
            }
            return
        }
        
        val currentStep = workoutManager.currentStep
        val exercise = currentStep?.exercise

        // TTS Voice cues logic
        if (exercise != null) {
            if (exercise.name != lastExerciseName) {
                lastExerciseName = exercise.name
                lastReps = 0
                if (exercise.name == "REST") {
                    ttsHelper?.speak("Take a rest break.", overrideCooldown = true)
                } else {
                    ttsHelper?.speak("Start ${exercise.name} exercise. Target is ${currentStep.targetReps} reps.", overrideCooldown = true)
                }
            } else {
                if (exercise.name == "REST") {
                    if (feedback.startsWith("Rest for")) {
                        ttsHelper?.speak(feedback, overrideCooldown = true)
                    } else if (feedback == "Rest finished!") {
                        ttsHelper?.speak("Rest finished! Get ready.", overrideCooldown = true)
                    }
                } else {
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
                        if (!rawFeedback.startsWith("Rep ") && !rawFeedback.startsWith("Remaining: ")) {
                            ttsHelper?.speak(rawFeedback, overrideCooldown = false)
                        }
                    }
                }
            }
        }

        // Convert to draw on screen
        val points = poseLandmarks.map { Pair(it.x(), it.y()) }

        runOnUiThread {
            overlayView.updateResults(
                points,
                exercise?.name ?: "Finished",
                reps,
                feedback,
                exercise?.color ?: android.graphics.Color.GREEN,
                exercise?.lastRepScore ?: -1,
                exercise?.lastRepMetrics
            )
        }
    }

    @Volatile private var isFinishing = false

    private fun finishWorkout() {
        if (isFinishing) return
        isFinishing = true

        val durationSeconds = ((System.currentTimeMillis() - workoutStartTime) / 1000).toInt()
        
        // Collect metrics
        val steps = workoutManager.steps
        var totalReps = 0
        val allScores = mutableListOf<Int>()
        val allHistory = mutableListOf<com.example.fitnesstrackerapp.logic.RepMetrics>()
        
        steps.forEach { step ->
            if (!step.isRest) {
                totalReps += step.exercise.repetitions
                allScores.addAll(step.exercise.repHistory.map { it.formScore })
                allHistory.addAll(step.exercise.repHistory)
            }
        }
        
        val avgScore = if (allScores.isNotEmpty()) allScores.average().toInt() else 85
        val bestScore = if (allScores.isNotEmpty()) allScores.maxOrNull() ?: 0 else 0
        
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid != null) {
            FirebaseFirestore.getInstance().collection("users").document(uid).get()
                .addOnSuccessListener { doc ->
                    val weight = doc.getDouble("weightKg") ?: 70.0
                    saveAndNavigate(uid, weight, durationSeconds, totalReps, avgScore, bestScore, allScores, allHistory)
                }
                .addOnFailureListener {
                    saveAndNavigate(uid, 70.0, durationSeconds, totalReps, avgScore, bestScore, allScores, allHistory)
                }
        } else {
            saveAndNavigate(null, 70.0, durationSeconds, totalReps, avgScore, bestScore, allScores, allHistory)
        }
    }

    private fun saveAndNavigate(
        uid: String?,
        weight: Double,
        durationSeconds: Int,
        totalReps: Int,
        avgScore: Int,
        bestScore: Int,
        scoresList: List<Int>,
        history: List<com.example.fitnesstrackerapp.logic.RepMetrics>
    ) {
        val met = 6.0
        val caloriesBurned = met * 3.5 * weight / 200.0 * (durationSeconds / 60.0)
        val roundedCalories = (kotlin.math.round(caloriesBurned * 10) / 10.0)

        // Calculate cadence stability (Standard Deviation of rep durations)
        var stdDev = 0.0
        var cadenceScore = 100.0
        if (history.size >= 2) {
            val repTimes = history.map { (it.eccentricDurationMs + it.concentricDurationMs).toDouble() }
            val mean = repTimes.average()
            val variance = repTimes.map { (it - mean) * (it - mean) }.sum() / repTimes.size
            stdDev = kotlin.math.sqrt(variance)
            // 10000ms = 10s deviation yields 0 score, 0ms deviation yields 100 score
            cadenceScore = ((10000.0 - stdDev) / 100.0).coerceIn(0.0, 100.0)
        }
        val roundedCadenceScore = (kotlin.math.round(cadenceScore * 10) / 10.0)
        val roundedStdDev = (kotlin.math.round(stdDev * 10) / 10.0)
        val volume = totalReps * weight

        val avgEcc = if (history.isNotEmpty()) history.map { it.eccentricDurationMs }.average().toLong() else 0L
        val avgConc = if (history.isNotEmpty()) history.map { it.concentricDurationMs }.average().toLong() else 0L

        val workoutData = hashMapOf(
            "date" to Timestamp(Date()),
            "workoutName" to "Mini Plan (Squat, Rest, Lunge)",
            "durationSeconds" to durationSeconds,
            "caloriesBurned" to roundedCalories,
            "totalReps" to totalReps,
            "averageFormScore" to avgScore,
            "weightKg" to weight,
            "volume" to volume,
            "avgEccentricDurationMs" to avgEcc,
            "avgConcentricDurationMs" to avgConc,
            "cadenceStability" to roundedStdDev,
            "cadenceScore" to roundedCadenceScore
        )

        val navigateAction = {
            val intent = Intent(this, ResultActivity::class.java).apply {
                putExtra(ResultActivity.EXTRA_NAME, "Mini Plan (Squat, Rest, Lunge)")
                putExtra(ResultActivity.EXTRA_TIME_BASED, false)
                putExtra(ResultActivity.EXTRA_ACHIEVED, totalReps)
                putExtra(ResultActivity.EXTRA_TARGET, 20)
                putExtra(ResultActivity.EXTRA_AVG_SCORE, avgScore)
                putExtra(ResultActivity.EXTRA_BEST, bestScore)
                putExtra(ResultActivity.EXTRA_SCORES, scoresList.toIntArray())
                putExtra(ResultActivity.EXTRA_AVG_ECC, avgEcc)
                putExtra(ResultActivity.EXTRA_AVG_CONC, avgConc)
            }
            startActivity(intent)
            finish()
        }

        if (uid != null) {
            val db = FirebaseFirestore.getInstance()
            db.collection("users").document(uid).collection("workouts").add(workoutData)
                .addOnSuccessListener {
                    val userRef = db.collection("users").document(uid)
                    db.runTransaction { tx ->
                        val snapshot = tx.get(userRef)
                        
                        // Core progression
                        val currentXp = snapshot.getLong("xpPoints")?.toInt() ?: 0
                        val newXp = currentXp + (totalReps * 10) + (avgScore / 2)
                        val newLvl = 1 + (newXp / 1000)
                        
                        // Lifetime stats aggregation
                        val currentTotalKcal = snapshot.getDouble("totalKcal") ?: 0.0
                        val currentTotalReps = snapshot.getLong("totalReps")?.toInt() ?: 0
                        val currentTotalWorkouts = snapshot.getLong("totalWorkouts")?.toInt() ?: 0
                        val currentOverallCadence = snapshot.getDouble("overallCadenceStability") ?: 0.0
                        
                        val newTotalKcal = currentTotalKcal + roundedCalories
                        val newTotalReps = currentTotalReps + totalReps
                        val newTotalWorkouts = currentTotalWorkouts + 1
                        val newOverallCadence = if (currentTotalWorkouts == 0) {
                            roundedCadenceScore
                        } else {
                            (currentOverallCadence * currentTotalWorkouts + roundedCadenceScore) / newTotalWorkouts
                        }
                        
                        // Weekly stats aggregation with calendar reset check
                        val now = Date()
                        val lastResetVal = snapshot.get("lastWeeklyReset")
                        var lastResetDate: Date? = null
                        if (lastResetVal is Timestamp) {
                            lastResetDate = lastResetVal.toDate()
                        } else if (lastResetVal is Long) {
                            lastResetDate = Date(lastResetVal)
                        } else if (lastResetVal is Date) {
                            lastResetDate = lastResetVal
                        }
                        
                        val shouldResetWeekly = if (lastResetDate == null) {
                            true
                        } else {
                            val calNow = java.util.Calendar.getInstance().apply { time = now }
                            val calLast = java.util.Calendar.getInstance().apply { time = lastResetDate }
                            calNow.get(java.util.Calendar.YEAR) != calLast.get(java.util.Calendar.YEAR) ||
                            calNow.get(java.util.Calendar.WEEK_OF_YEAR) != calLast.get(java.util.Calendar.WEEK_OF_YEAR)
                        }
                        
                        val newWeeklyKcal: Double
                        val newWeeklyWorkouts: Int
                        val newWeeklyCadence: Double
                        val newWeeklyReset: Timestamp
                        
                        if (shouldResetWeekly) {
                            newWeeklyKcal = roundedCalories
                            newWeeklyWorkouts = 1
                            newWeeklyCadence = roundedCadenceScore
                            newWeeklyReset = Timestamp(now)
                        } else {
                            val currentWeeklyKcal = snapshot.getDouble("weeklyKcal") ?: 0.0
                            val currentWeeklyWorkouts = snapshot.getLong("weeklyWorkouts")?.toInt() ?: 0
                            val currentWeeklyCadence = snapshot.getDouble("weeklyCadenceStability") ?: 0.0
                            
                            newWeeklyKcal = currentWeeklyKcal + roundedCalories
                            newWeeklyWorkouts = currentWeeklyWorkouts + 1
                            newWeeklyCadence = (currentWeeklyCadence * currentWeeklyWorkouts + roundedCadenceScore) / newWeeklyWorkouts
                            newWeeklyReset = if (lastResetVal is Timestamp) lastResetVal else Timestamp(now)
                        }
                        
                        // Execute transaction updates
                        tx.update(userRef, "xpPoints", newXp)
                        tx.update(userRef, "level", newLvl)
                        tx.update(userRef, "totalKcal", newTotalKcal)
                        tx.update(userRef, "totalReps", newTotalReps)
                        tx.update(userRef, "totalWorkouts", newTotalWorkouts)
                        tx.update(userRef, "overallCadenceStability", newOverallCadence)
                        tx.update(userRef, "weeklyKcal", newWeeklyKcal)
                        tx.update(userRef, "weeklyWorkouts", newWeeklyWorkouts)
                        tx.update(userRef, "weeklyCadenceStability", newWeeklyCadence)
                        tx.update(userRef, "lastWeeklyReset", newWeeklyReset)
                        tx.update(userRef, "lastActive", Timestamp(now))
                    }.addOnCompleteListener {
                        navigateAction()
                    }
                }
                .addOnFailureListener {
                    navigateAction()
                }
        } else {
            navigateAction()
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
        ttsHelper?.shutdown()
    }

    companion object {
        private const val TAG = "FitnessTracker"
    }
}