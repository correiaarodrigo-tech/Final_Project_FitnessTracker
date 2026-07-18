package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*
import org.json.JSONArray
import org.json.JSONObject

class StartPlanActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        enableEdgeToEdge()

        val planName = intent.getStringExtra("EXTRA_PLAN_NAME") ?: "Mini Plano: Agachamento, Flexão, Afundo"
        val stepsJson = intent.getStringExtra("EXTRA_PLAN_STEPS_JSON") ?: getDefaultStepsJson()

        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    PlanOverviewScreen(
                        planName = planName,
                        stepsJson = stepsJson,
                        onStartWorkout = {
                            val intent = Intent(this@StartPlanActivity, MainActivity::class.java).apply {
                                putExtra("EXTRA_PLAN_NAME", planName)
                                putExtra("EXTRA_PLAN_STEPS_JSON", stepsJson)
                            }
                            startActivity(intent)
                            finish()
                        },
                        onBack = { finish() }
                    )
                }
            }
        }
    }

    private fun getDefaultStepsJson(): String {
        return JSONArray().apply {
            put(JSONObject().apply { put("type", "SQUAT"); put("value", 10) })
            put(JSONObject().apply { put("type", "REST"); put("value", 15) })
            put(JSONObject().apply { put("type", "PUSHUP"); put("value", 10) })
            put(JSONObject().apply { put("type", "REST"); put("value", 15) })
            put(JSONObject().apply { put("type", "LUNGE"); put("value", 10) })
        }.toString()
    }
}

private data class ParsedStep(val type: String, val value: Int)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlanOverviewScreen(
    planName: String,
    stepsJson: String,
    onStartWorkout: () -> Unit,
    onBack: () -> Unit
) {
    // Parse steps
    val parsedSteps = remember(stepsJson) {
        val list = mutableListOf<ParsedStep>()
        try {
            val arr = JSONArray(stepsJson)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                list.add(ParsedStep(obj.getString("type"), obj.getInt("value")))
            }
        } catch (e: Exception) {
            // Fallback
            list.add(ParsedStep("SQUAT", 10))
            list.add(ParsedStep("REST", 15))
            list.add(ParsedStep("LUNGE", 10))
        }
        list
    }

    // Dynamic stats calculations
    val totalDurationSecs = remember(parsedSteps) {
        parsedSteps.sumOf { step ->
            if (step.type == "REST") step.value else step.value * 3 // approx 3s per rep
        }
    }
    val averageMet = remember(parsedSteps) {
        val totalMetWeighted = parsedSteps.sumOf { step ->
            val met = when (step.type) {
                "SQUAT" -> 6.0
                "PUSHUP" -> 4.0
                "LUNGE" -> 6.0
                else -> 1.0 // REST
            }
            val weight = if (step.type == "REST") step.value else step.value * 3
            met * weight
        }
        if (totalDurationSecs > 0) totalMetWeighted / totalDurationSecs else 1.0
    }
    val totalCalories = remember(totalDurationSecs, averageMet) {
        // Formula: Kcal = (DurationHours) * MET * 75kg
        ((totalDurationSecs.toFloat() / 3600f) * averageMet * 75f).toInt().coerceAtLeast(1)
    }

    var showDisclaimer by remember { mutableStateOf(false) }

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
                    text = "← Voltar",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = PrimaryCyan,
                    modifier = Modifier
                        .clickable(onClick = onBack)
                        .padding(8.dp)
                )
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Plano de Treino",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Spacer(modifier = Modifier.weight(1f))
                Spacer(modifier = Modifier.width(48.dp))
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Plan Title
            Text(
                text = planName,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Plan Badges Row
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                val mins = totalDurationSecs / 60
                val secs = totalDurationSecs % 60
                val durationStr = if (mins > 0) "~$mins min $secs s" else "~$secs s"
                BadgeItem(text = durationStr, color = PrimaryCyan)
                BadgeItem(text = "MET %.1f".format(averageMet), color = SecondaryPurple)
                BadgeItem(text = "~$totalCalories kcal", color = AccentGreen)
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Plan Description Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                border = BorderStroke(1.dp, BorderMuted)
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text(
                        text = "Descrição do Exercício",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = "Um treino cardiovascular e muscular personalizado e adaptado às tuas capacidades. Siga as instruções no ecrã e utilize o feedback por voz para otimizar o seu desempenho e evitar lesões.",
                        fontSize = 13.sp,
                        color = TextSecondary,
                        lineHeight = 18.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Exercise Steps Title
            Text(
                text = "Passos do Treino (${parsedSteps.size})",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                textAlign = TextAlign.Start
            )

            // Step List
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                parsedSteps.forEachIndexed { index, step ->
                    val isRest = step.type == "REST"
                    val title = when (step.type) {
                        "SQUAT" -> "Agachamentos (Squats)"
                        "PUSHUP" -> "Flexões (Push-Ups)"
                        "LUNGE" -> "Afundos (Lunges)"
                        else -> "Recuperação Cardiovascular"
                    }
                    val detail = if (isRest) "${step.value} Segundos" else "${step.value} Repetições"
                    val desc = when (step.type) {
                        "SQUAT" -> "Fortalecimento de quadríceps e glúteos. Agache até as coxas ficarem paralelas ao chão."
                        "PUSHUP" -> "Exercício peitoral e tríceps. Mantenha as costas retas e o core bem ativado."
                        "LUNGE" -> "Trabalho de equilíbrio e força das pernas. Passada alternada focando na profundidade."
                        else -> "Descanse para recuperar o ritmo cardíaco antes do próximo exercício."
                    }
                    val color = if (isRest) Color.Gray else if (step.type == "PUSHUP") AccentGreen else PrimaryCyan

                    WorkoutStepCard(
                        stepNumber = "${index + 1}",
                        title = title,
                        detail = detail,
                        description = desc,
                        accentColor = color
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Start Plan Button
            Button(
                onClick = { showDisclaimer = true },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(58.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
                contentPadding = PaddingValues()
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            brush = Brush.horizontalGradient(
                                colors = listOf(PrimaryCyan, SecondaryPurple)
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "INICIAR TREINO",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0C0F14),
                        letterSpacing = 1.5.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }

    // Disclaimer & Calibration Confirmation Dialog
    if (showDisclaimer) {
        val exercisesOnly = parsedSteps.filter { it.type != "REST" }
        val squatsCount = exercisesOnly.count { it.type == "SQUAT" }
        val pushupsCount = exercisesOnly.count { it.type == "PUSHUP" }
        val lungesCount = exercisesOnly.count { it.type == "LUNGE" }

        AlertDialog(
            onDismissRequest = { showDisclaimer = false },
            title = {
                Text(
                    text = "Calibração & Preparação",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    color = Color.White
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        text = "Vais iniciar o treino com as seguintes metas:",
                        color = TextSecondary,
                        fontSize = 13.sp
                    )
                    if (squatsCount > 0) Text("  ${stringResource(R.string.plan_squats_count, squatsCount)}", color = Color.White, fontSize = 13.sp)
                    if (pushupsCount > 0) Text("  ${stringResource(R.string.plan_pushups_count, pushupsCount)}", color = Color.White, fontSize = 13.sp)
                    if (lungesCount > 0) Text("  ${stringResource(R.string.plan_lunges_count, lungesCount)}", color = Color.White, fontSize = 13.sp)

                    Spacer(modifier = Modifier.height(6.dp))

                    Text(
                        text = "Aviso de Distância:\n" +
                               "Para garantir a precisão da IA, coloca o teu telemóvel a uma distância sugerida de 2 a 6 metros do teu corpo. Garanta que o teu corpo inteiro (da cabeça aos pés) esteja visível na câmara.",
                        color = PrimaryCyan,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDisclaimer = false
                        onStartWorkout()
                    }
                ) {
                    Text(stringResource(R.string.btn_start_uppercase), color = PrimaryCyan, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDisclaimer = false }) {
                    Text(stringResource(R.string.btn_cancel_uppercase), color = Color.Gray)
                }
            },
            containerColor = DarkSurface,
            shape = RoundedCornerShape(16.dp)
        )
    }
}

@Composable
fun BadgeItem(text: String, color: Color) {
    Box(
        modifier = Modifier
            .background(color.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
            .border(1.dp, color.copy(alpha = 0.4f), RoundedCornerShape(6.dp))
            .padding(horizontal = 10.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            color = color,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun WorkoutStepCard(
    stepNumber: String,
    title: String,
    detail: String,
    description: String,
    accentColor: Color
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Step Number Circle
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(accentColor.copy(alpha = 0.15f), RoundedCornerShape(20.dp))
                    .border(1.5.dp, accentColor, RoundedCornerShape(20.dp)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = stepNumber,
                    color = accentColor,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.width(16.dp))

            // Step Details
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = title,
                        color = Color.White,
                        fontSize = 15.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = detail,
                        color = accentColor,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = description,
                    color = TextSecondary,
                    fontSize = 11.sp,
                    lineHeight = 15.sp
                )
            }
        }
    }
}
