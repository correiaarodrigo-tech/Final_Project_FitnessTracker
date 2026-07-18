package com.example.fitnesstrackerapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*

/**
 * Post-exercise summary screen. Receives a session summary via Intent extras and
 * renders it in the app's dark/neon aesthetic.
 */
class ResultActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val name = intent.getStringExtra(EXTRA_NAME) ?: "Exercise"
        val timeBased = intent.getBooleanExtra(EXTRA_TIME_BASED, false)
        val achieved = intent.getIntExtra(EXTRA_ACHIEVED, 0)
        val targetVal = intent.getIntExtra(EXTRA_TARGET, 20)
        val avgScore = intent.getIntExtra(EXTRA_AVG_SCORE, 0)
        val avgEcc = intent.getLongExtra(EXTRA_AVG_ECC, 0L)
        val avgConc = intent.getLongExtra(EXTRA_AVG_CONC, 0L)
        val best = intent.getIntExtra(EXTRA_BEST, 0)
        val scores = intent.getIntArrayExtra(EXTRA_SCORES)?.toList() ?: emptyList()

        setContent {
            FitnessTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ResultScreen(
                        name = name,
                        timeBased = timeBased,
                        achieved = achieved,
                        target = targetVal,
                        avgScore = avgScore,
                        avgEcc = avgEcc,
                        avgConc = avgConc,
                        best = best,
                        scores = scores,
                        onDone = { finish() }
                    )
                }
            }
        }
    }

    companion object {
        const val EXTRA_NAME = "name"
        const val EXTRA_TIME_BASED = "timeBased"
        const val EXTRA_ACHIEVED = "achieved"
        const val EXTRA_TARGET = "target"
        const val EXTRA_AVG_SCORE = "avgScore"
        const val EXTRA_AVG_ECC = "avgEcc"
        const val EXTRA_AVG_CONC = "avgConc"
        const val EXTRA_BEST = "best"
        const val EXTRA_SCORES = "scores"
    }
}

private fun scoreColor(score: Int): Color = when {
    score >= 80 -> AccentGreen
    score >= 60 -> Color(0xFFFFD23F)
    else -> Color(0xFFFF4D4D)
}

@Composable
private fun ResultScreen(
    name: String,
    timeBased: Boolean,
    achieved: Int,
    target: Int,
    avgScore: Int,
    avgEcc: Long,
    avgConc: Long,
    best: Int,
    scores: List<Int>,
    onDone: () -> Unit
) {
    // For holds the ring shows completion; for reps it shows average form.
    val ringValue = if (timeBased) ((achieved * 100) / target.coerceAtLeast(1)) else avgScore
    val ringLabel = if (timeBased) "Completed" else "Avg Form"
    val ringColor = if (timeBased) PrimaryCyan else scoreColor(avgScore)

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
            Spacer(Modifier.height(32.dp))

            Text(
                text = "WORKOUT COMPLETE",
                fontSize = 14.sp,
                letterSpacing = 3.sp,
                fontWeight = FontWeight.Bold,
                color = AccentGreen
            )
            Spacer(Modifier.height(6.dp))
            Text(
                text = name,
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            Spacer(Modifier.height(28.dp))

            // Score / completion ring
            ScoreRing(value = ringValue, label = ringLabel, color = ringColor)

            Spacer(Modifier.height(28.dp))

            // Stat cards
            if (timeBased) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    StatCard(Modifier.weight(1f), "Time held", "${achieved}s", PrimaryCyan)
                    StatCard(Modifier.weight(1f), "Goal", "${target}s", SecondaryPurple)
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    StatCard(Modifier.weight(1f), "Reps", "$achieved/$target", PrimaryCyan)
                    StatCard(Modifier.weight(1f), "Best rep", "$best", AccentGreen)
                }
                Spacer(Modifier.height(16.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    StatCard(Modifier.weight(1f), "Avg lowering", "${avgEcc} ms", SecondaryPurple)
                    StatCard(Modifier.weight(1f), "Avg lifting", "${avgConc} ms", PrimaryCyan)
                }
            }

            // Per-rep form chart
            if (scores.isNotEmpty()) {
                Spacer(Modifier.height(24.dp))
                RepScoreChart(scores)
            }

            Spacer(Modifier.height(32.dp))

            Button(
                onClick = onDone,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan)
            ) {
                Text(stringResource(R.string.btn_done), color = Color(0xFF0C0F14), fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }

            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
private fun ScoreRing(value: Int, label: String, color: Color) {
    Box(contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(190.dp)) {
            val stroke = 22.dp.toPx()
            val inset = stroke / 2f
            val arcSize = Size(size.width - stroke, size.height - stroke)
            // Track
            drawArc(
                color = BorderMuted,
                startAngle = -90f,
                sweepAngle = 360f,
                useCenter = false,
                topLeft = androidx.compose.ui.geometry.Offset(inset, inset),
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round)
            )
            // Value
            drawArc(
                color = color,
                startAngle = -90f,
                sweepAngle = 360f * (value.coerceIn(0, 100) / 100f),
                useCenter = false,
                topLeft = androidx.compose.ui.geometry.Offset(inset, inset),
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round)
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("$value", fontSize = 52.sp, fontWeight = FontWeight.Bold, color = Color.White)
            Text(label, fontSize = 13.sp, color = TextSecondary, fontWeight = FontWeight.Medium)
        }
    }
}

@Composable
private fun StatCard(modifier: Modifier, label: String, value: String, accent: Color) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(label, color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(6.dp))
            Text(value, color = accent, fontSize = 24.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun RepScoreChart(scores: List<Int>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(stringResource(R.string.form_per_rep), color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(14.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(90.dp),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                scores.forEach { s ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight(fraction = (s.coerceIn(0, 100) / 100f).coerceAtLeast(0.04f))
                            .clip(RoundedCornerShape(3.dp))
                            .background(scoreColor(s))
                    )
                }
            }
        }
    }
}
