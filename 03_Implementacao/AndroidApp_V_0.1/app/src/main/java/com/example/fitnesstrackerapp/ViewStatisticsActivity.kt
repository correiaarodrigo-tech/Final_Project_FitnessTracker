package com.example.fitnesstrackerapp

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*
import com.google.firebase.Timestamp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import java.text.SimpleDateFormat
import java.util.*

class ViewStatisticsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge()
        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    StatisticsScreen(onBack = { finish() })
                }
            }
        }
    }
}

data class WorkoutLog(
    val date: Any? = null,
    val workoutName: String = "",
    val durationSeconds: Int = 0,
    val caloriesBurned: Double = 0.0,
    val totalReps: Int = 0,
    val averageFormScore: Int = 0
) {
    fun getDateAsDate(): Date {
        return when (date) {
            is Timestamp -> date.toDate()
            is Date -> date
            is Long -> Date(date)
            is Double -> Date(date.toLong())
            is Map<*, *> -> {
                val seconds = date["seconds"] as? Long ?: 0L
                val nanoseconds = date["nanoseconds"] as? Long ?: 0L
                Date(seconds * 1000 + (nanoseconds / 1000000))
            }
            else -> Date()
        }
    }
}

@Composable
fun StatisticsScreen(onBack: () -> Unit) {
    val auth = remember { FirebaseAuth.getInstance() }
    val db = remember { FirebaseFirestore.getInstance() }
    val currentUser = auth.currentUser
    val context = LocalContext.current

    var workouts by remember { mutableStateOf<List<WorkoutLog>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var isSavingSeed by remember { mutableStateOf(false) }
    var isClearing by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    fun fetchWorkouts() {
        val uid = currentUser?.uid
        if (uid == null) {
            errorMessage = "User not logged in."
            isLoading = false
            return
        }

        isLoading = true
        db.collection("users").document(uid).collection("workouts")
            .orderBy("date", Query.Direction.DESCENDING)
            .get()
            .addOnSuccessListener { querySnapshot ->
                val list = querySnapshot.documents.mapNotNull { doc ->
                    try {
                        doc.toObject(WorkoutLog::class.java)
                    } catch (e: Exception) {
                        null
                    }
                }
                if (list.isEmpty()) {
                    // Auto-seed if empty
                    isSavingSeed = true
                    seedMockData(uid, db) { seededList ->
                        workouts = seededList
                        isSavingSeed = false
                        isLoading = false
                        Toast.makeText(context, "Seeded 5 sample workouts for testing!", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    workouts = list
                    isLoading = false
                }
            }
            .addOnFailureListener { e ->
                errorMessage = e.localizedMessage
                isLoading = false
            }
    }

    LaunchedEffect(currentUser) {
        fetchWorkouts()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 540.dp)
                .fillMaxWidth()
                .padding(horizontal = 24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "← Back",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = PrimaryCyan,
                    modifier = Modifier
                        .clickable(onClick = onBack)
                        .padding(8.dp)
                )
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Performance Stats",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Spacer(modifier = Modifier.weight(1f))
                Spacer(modifier = Modifier.width(48.dp))
            }

            if (isLoading || isSavingSeed || isClearing) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = PrimaryCyan)
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = if (isSavingSeed) "Seeding sample workouts..." else if (isClearing) "Clearing history..." else "Loading statistics...",
                            color = TextSecondary,
                            fontSize = 14.sp
                        )
                    }
                }
            } else if (errorMessage != null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Error: $errorMessage",
                        color = Color.Red,
                        fontSize = 14.sp,
                        textAlign = TextAlign.Center
                    )
                }
            } else {
                Spacer(modifier = Modifier.height(8.dp))

                // Custom Canvas Chart
                RepsCanvasChart(workouts = workouts)

                Spacer(modifier = Modifier.height(24.dp))

                // Stats Cards
                StatsSummaryGrid(workouts = workouts)

                Spacer(modifier = Modifier.height(24.dp))

                // History List
                HistoryList(workouts = workouts)

                Spacer(modifier = Modifier.height(32.dp))

                // Developer actions
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Button(
                        onClick = {
                            val uid = currentUser?.uid
                            if (uid != null) {
                                isClearing = true
                                clearHistory(uid, db) {
                                    workouts = emptyList()
                                    isClearing = false
                                    Toast.makeText(context, "Workout history cleared!", Toast.LENGTH_SHORT).show()
                                }
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF4D4D).copy(alpha = 0.15f)),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, Color(0xFFFF4D4D).copy(alpha = 0.4f))
                    ) {
                        Text("Clear History", color = Color(0xFFFF4D4D), fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }

                    Button(
                        onClick = {
                            val uid = currentUser?.uid
                            if (uid != null) {
                                isSavingSeed = true
                                seedMockData(uid, db) { seededList ->
                                    workouts = seededList
                                    isSavingSeed = false
                                    Toast.makeText(context, "Mock data seeded successfully!", Toast.LENGTH_SHORT).show()
                                }
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan.copy(alpha = 0.15f)),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, PrimaryCyan.copy(alpha = 0.4f))
                    ) {
                        Text("Seed Mock Data", color = PrimaryCyan, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                }

                Spacer(modifier = Modifier.height(48.dp))
            }
        }
    }
}

@Composable
fun RepsCanvasChart(workouts: List<WorkoutLog>) {
    // Generate the last 7 calendar days
    val last7Days = remember {
        (0..6).map { i ->
            val cal = Calendar.getInstance()
            cal.add(Calendar.DAY_OF_YEAR, -i)
            cal
        }.reversed()
    }

    val sdfLabel = remember { SimpleDateFormat("EEE", Locale.getDefault()) }
    val sdfKey = remember { SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()) }

    // Aggregate reps for each of the last 7 days
    val repsData = remember(workouts) {
        last7Days.map { cal ->
            val key = sdfKey.format(cal.time)
            val repsSum = workouts.filter { w ->
                sdfKey.format(w.getDateAsDate()) == key
            }.sumOf { it.totalReps }
            Pair(sdfLabel.format(cal.time), repsSum)
        }
    }

    val maxReps = remember(repsData) {
        repsData.map { it.second }.maxOrNull()?.coerceAtLeast(10) ?: 10
    }

    var animateTrigger by remember { mutableStateOf(false) }
    LaunchedEffect(workouts) {
        animateTrigger = true
    }
    val animationScale by animateFloatAsState(
        targetValue = if (animateTrigger) 1f else 0f,
        animationSpec = tween(durationMillis = 1000, easing = FastOutSlowInEasing)
    )

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Weekly Activity (Reps)",
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(20.dp))

            val density = LocalDensity.current
            val textPaint = remember(density) {
                android.graphics.Paint().apply {
                    color = android.graphics.Color.WHITE
                    textAlign = android.graphics.Paint.Align.CENTER
                    textSize = with(density) { 11.sp.toPx() }
                    typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
                }
            }
            val labelPaint = remember(density) {
                android.graphics.Paint().apply {
                    color = android.graphics.Color.parseColor("#9EADB8") // TextSecondary
                    textAlign = android.graphics.Paint.Align.CENTER
                    textSize = with(density) { 10.sp.toPx() }
                }
            }

            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp)
            ) {
                val chartWidth = size.width
                val chartHeight = size.height - 30.dp.toPx()
                val chartBottom = size.height - 24.dp.toPx()
                val chartTop = 16.dp.toPx()

                // Draw 3 background grid lines (horizontal)
                val gridLines = listOf(chartTop, chartTop + chartHeight / 2, chartBottom)
                gridLines.forEach { y ->
                    drawLine(
                        color = BorderMuted.copy(alpha = 0.5f),
                        start = Offset(0f, y),
                        end = Offset(chartWidth, y),
                        strokeWidth = 1.dp.toPx()
                    )
                }

                val barCount = repsData.size
                val barSpacing = 16.dp.toPx()
                val totalSpacing = barSpacing * (barCount - 1)
                val barWidth = (chartWidth - totalSpacing) / barCount

                repsData.forEachIndexed { index, (dayLabel, reps) ->
                    val x = index * (barWidth + barSpacing)
                    val targetHeight = (reps.toFloat() / maxReps) * chartHeight
                    val animatedHeight = targetHeight * animationScale
                    val y = chartBottom - animatedHeight

                    // Draw bar with gradient
                    if (reps > 0) {
                        drawRoundRect(
                            brush = Brush.verticalGradient(
                                colors = listOf(PrimaryCyan, SecondaryPurple)
                            ),
                            topLeft = Offset(x, y),
                            size = Size(barWidth, animatedHeight),
                            cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
                        )
                    } else {
                        // Tiny placeholder indicator for zero reps day
                        drawRoundRect(
                            color = BorderMuted,
                            topLeft = Offset(x, chartBottom - 4.dp.toPx()),
                            size = Size(barWidth, 4.dp.toPx()),
                            cornerRadius = CornerRadius(2.dp.toPx(), 2.dp.toPx())
                        )
                    }

                    // Native text drawing
                    drawIntoCanvas { canvas ->
                        canvas.nativeCanvas.drawText(
                            dayLabel,
                            x + barWidth / 2,
                            chartBottom + 18.dp.toPx(),
                            labelPaint
                        )
                        if (reps > 0) {
                            canvas.nativeCanvas.drawText(
                                "$reps",
                                x + barWidth / 2,
                                y - 6.dp.toPx(),
                                textPaint
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun StatsSummaryGrid(workouts: List<WorkoutLog>) {
    val totalWorkouts = workouts.size
    val totalSeconds = workouts.sumOf { it.durationSeconds }
    val totalCalories = workouts.sumOf { it.caloriesBurned }
    val avgScore = if (workouts.isNotEmpty()) workouts.map { it.averageFormScore }.average().toInt() else 0

    val hours = totalSeconds / 3600
    val minutes = (totalSeconds % 3600) / 60
    val durationText = if (hours > 0) {
        "${hours}h ${minutes}m"
    } else {
        "${minutes} min"
    }

    Column(
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            StatsGridCard(
                modifier = Modifier.weight(1f),
                title = "Total Workouts",
                value = "$totalWorkouts",
                accentColor = PrimaryCyan,
                metric = "sessions"
            )
            StatsGridCard(
                modifier = Modifier.weight(1f),
                title = "Active Time",
                value = durationText,
                accentColor = SecondaryPurple,
                metric = "duration"
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            StatsGridCard(
                modifier = Modifier.weight(1f),
                title = "Calories Burned",
                value = String.format(Locale.US, "%.1f", totalCalories),
                accentColor = AccentGreen,
                metric = "kcal"
            )
            StatsGridCard(
                modifier = Modifier.weight(1f),
                title = "Average Form",
                value = "$avgScore%",
                accentColor = PrimaryCyan,
                metric = "evaluation"
            )
        }
    }
}

@Composable
fun StatsGridCard(
    modifier: Modifier,
    title: String,
    value: String,
    accentColor: Color,
    metric: String
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = title,
                color = TextSecondary,
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = value,
                    color = Color.White,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = metric,
                    color = accentColor,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 2.dp)
                )
            }
        }
    }
}

@Composable
fun HistoryList(workouts: List<WorkoutLog>) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = "Workout History",
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        if (workouts.isEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                border = BorderStroke(1.dp, BorderMuted)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No workouts recorded yet.",
                        color = TextSecondary,
                        fontSize = 13.sp
                    )
                }
            }
        } else {
            val sdfDate = remember { SimpleDateFormat("MMMM d, yyyy 'at' HH:mm", Locale.getDefault()) }
            
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                workouts.forEach { log ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(containerColor = DarkSurface),
                        border = BorderStroke(1.dp, BorderMuted)
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = log.workoutName,
                                    color = Color.White,
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Box(
                                    modifier = Modifier
                                        .background(PrimaryCyan.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                        .border(1.dp, PrimaryCyan.copy(alpha = 0.4f), RoundedCornerShape(6.dp))
                                        .padding(horizontal = 8.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = "${log.totalReps} reps",
                                        color = PrimaryCyan,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(4.dp))
                            
                            Text(
                                text = sdfDate.format(log.getDateAsDate()),
                                color = TextSecondary,
                                fontSize = 11.sp
                            )

                            Spacer(modifier = Modifier.height(10.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                Text(
                                    text = "Form: ${log.averageFormScore}%",
                                    color = if (log.averageFormScore >= 80) AccentGreen else if (log.averageFormScore >= 60) Color(0xFFFFD23F) else Color(0xFFFF4D4D),
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Duration: ${log.durationSeconds / 60}m ${log.durationSeconds % 60}s",
                                    color = TextSecondary,
                                    fontSize = 12.sp
                                )
                                Text(
                                    text = "Calories: ${log.caloriesBurned} kcal",
                                    color = TextSecondary,
                                    fontSize = 12.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun seedMockData(
    uid: String,
    db: FirebaseFirestore,
    onComplete: (List<WorkoutLog>) -> Unit
) {
    val mockList = mutableListOf<WorkoutLog>()
    val batch = db.batch()

    // 5 workouts across last 5 days
    val workoutsData = listOf(
        Pair(5, WorkoutLog(
            workoutName = "Mini Plan (Squat, Rest, Lunge)",
            durationSeconds = 95,
            caloriesBurned = 14.2,
            totalReps = 20,
            averageFormScore = 80
        )),
        Pair(4, WorkoutLog(
            workoutName = "Mini Plan (Squat, Rest, Lunge)",
            durationSeconds = 105,
            caloriesBurned = 15.8,
            totalReps = 20,
            averageFormScore = 83
        )),
        Pair(3, WorkoutLog(
            workoutName = "Mini Plan (Squat, Rest, Lunge)",
            durationSeconds = 120,
            caloriesBurned = 18.0,
            totalReps = 20,
            averageFormScore = 87
        )),
        Pair(2, WorkoutLog(
            workoutName = "Mini Plan (Squat, Rest, Lunge)",
            durationSeconds = 110,
            caloriesBurned = 16.5,
            totalReps = 20,
            averageFormScore = 85
        )),
        Pair(1, WorkoutLog(
            workoutName = "Mini Plan (Squat, Rest, Lunge)",
            durationSeconds = 130,
            caloriesBurned = 19.5,
            totalReps = 20,
            averageFormScore = 91
        ))
    )

    workoutsData.forEach { (daysAgo, log) ->
        val cal = Calendar.getInstance()
        cal.add(Calendar.DAY_OF_YEAR, -daysAgo)
        cal.set(Calendar.HOUR_OF_DAY, 9 + daysAgo)
        cal.set(Calendar.MINUTE, 20 * daysAgo)

        val timestamp = Timestamp(cal.time)
        val finalLog = log.copy(date = timestamp)
        mockList.add(finalLog)

        val docRef = db.collection("users").document(uid).collection("workouts").document()
        batch.set(docRef, finalLog)
    }

    batch.commit()
        .addOnSuccessListener {
            onComplete(mockList.sortedByDescending { it.getDateAsDate().time })
        }
        .addOnFailureListener {
            onComplete(emptyList())
        }
}

private fun clearHistory(
    uid: String,
    db: FirebaseFirestore,
    onComplete: () -> Unit
) {
    db.collection("users").document(uid).collection("workouts")
        .get()
        .addOnSuccessListener { querySnapshot ->
            val batch = db.batch()
            querySnapshot.documents.forEach { doc ->
                batch.delete(doc.reference)
            }
            batch.commit()
                .addOnSuccessListener {
                    onComplete()
                }
                .addOnFailureListener {
                    onComplete()
                }
        }
        .addOnFailureListener {
            onComplete()
        }
}
