package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*
import com.google.firebase.Timestamp
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import org.json.JSONArray
import org.json.JSONObject
import java.util.Date

class CreatePlanActivity : ComponentActivity() {
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
                    CreatePlanScreen(
                        onBack = { finish() },
                        onSaved = { planName, stepsJson ->
                            val intent = Intent(this, StartPlanActivity::class.java).apply {
                                putExtra("EXTRA_PLAN_NAME", planName)
                                putExtra("EXTRA_PLAN_STEPS_JSON", stepsJson)
                                putExtra("EXTRA_IS_CUSTOM", true)
                            }
                            startActivity(intent)
                            finish()
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CreatePlanScreen(onBack: () -> Unit, onSaved: (String, String) -> Unit) {
    val context = LocalContext.current
    var planName by remember { mutableStateOf("") }
    
    // Each step is represented as: mapOf("type" to "SQUAT"/"PUSHUP"/"LUNGE"/"REST", "value" to 10/30)
    val steps = remember { mutableStateListOf<Map<String, Any>>() }

    // Step Creator Panel State
    var selectedType by remember { mutableStateOf("SQUAT") }
    var currentVal by remember { mutableStateOf(10) }

    // Update value when exercise type changes to fit the default ranges
    LaunchedEffect(selectedType) {
        currentVal = when (selectedType) {
            "SQUAT" -> 10
            "PUSHUP" -> 8
            "LUNGE" -> 10
            else -> 30 // REST
        }
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
                    text = "← Cancel",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = PrimaryCyan,
                    modifier = Modifier
                        .clickable(onClick = onBack)
                        .padding(8.dp)
                )
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = "Criar Plano",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Spacer(modifier = Modifier.weight(1f))
                Spacer(modifier = Modifier.width(56.dp))
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Plan Name Input Field
            OutlinedTextField(
                value = planName,
                onValueChange = { planName = it },
                label = { Text(stringResource(R.string.plan_name_label), color = TextSecondary) },
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedBorderColor = PrimaryCyan,
                    unfocusedBorderColor = BorderMuted,
                    cursorColor = PrimaryCyan
                ),
                shape = RoundedCornerShape(12.dp),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Add Step Selection Box
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = DarkSurface),
                border = BorderStroke(1.dp, BorderMuted)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Adicionar Passo",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Spacer(modifier = Modifier.height(12.dp))

                    // Exercise Toggle selectors
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color.White.copy(alpha = 0.03f), RoundedCornerShape(8.dp))
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        listOf("SQUAT" to "Agach.", "PUSHUP" to "Flexão", "LUNGE" to "Afundo", "REST" to "Desc.").forEach { (type, label) ->
                            val active = selectedType == type
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(6.dp))
                                    .clickable { selectedType = type }
                                    .background(if (active) PrimaryCyan.copy(alpha = 0.15f) else Color.Transparent)
                                    .padding(vertical = 8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = label,
                                    color = if (active) PrimaryCyan else TextSecondary,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Reps/Seconds Scroller Control
                    val range = when (selectedType) {
                        "SQUAT" -> 5..25
                        "PUSHUP" -> 3..15
                        "LUNGE" -> 5..20
                        else -> 30..120
                    }
                    val unit = if (selectedType == "REST") "segundos" else if (selectedType == "LUNGE") "reps (cada perna)" else "reps"

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Quantidade: $currentVal $unit",
                            color = TextSecondary,
                            fontSize = 13.sp
                        )
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            IconButton(
                                onClick = { if (currentVal > range.first) currentVal -= if (selectedType == "REST") 5 else 1 },
                                modifier = Modifier.background(Color.White.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                            ) {
                                Text("-", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                            }
                            Text("$currentVal", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                            IconButton(
                                onClick = { if (currentVal < range.last) currentVal += if (selectedType == "REST") 5 else 1 },
                                modifier = Modifier.background(Color.White.copy(alpha = 0.05f), RoundedCornerShape(8.dp))
                            ) {
                                Text("+", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // Insert Step Button
                    Button(
                        onClick = {
                            val newStep = mapOf("type" to selectedType, "value" to currentVal)
                            
                            // Check if last step was an exercise, and we are adding another exercise
                            if (steps.isNotEmpty() && selectedType != "REST") {
                                val lastType = steps.last()["type"] as String
                                if (lastType != "REST") {
                                    // Auto-insert a 30s rest step in between exercises to enforce rest constraints
                                    steps.add(mapOf("type" to "REST", "value" to 30))
                                    Toast.makeText(context, context.getString(R.string.toast_rest_added), Toast.LENGTH_SHORT).show()
                                }
                            }
                            
                            steps.add(newStep)
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Text(stringResource(R.string.btn_add_to_workout), color = Color.Black, fontWeight = FontWeight.Bold)
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Current plan steps preview title
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Exercícios no Plano (${steps.size})",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                if (steps.isNotEmpty()) {
                    Text(
                        text = "Limpar Tudo",
                        color = Color(0xFFFF4D4D),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.clickable { steps.clear() }
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Steps preview list
            if (steps.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(100.dp)
                        .border(1.dp, BorderMuted, RoundedCornerShape(12.dp))
                        .padding(16.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(stringResource(R.string.add_exercises_hint), color = TextSecondary, fontSize = 13.sp)
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    steps.forEachIndexed { index, step ->
                        val type = step["type"] as String
                        val value = step["value"] as Int
                        val isExercise = type != "REST"
                        val stepTitle = when (type) {
                            "SQUAT" -> "Agachamentos (Squats)"
                            "PUSHUP" -> "Flexões (Push-Ups)"
                            "LUNGE" -> "Afundos (Lunges)"
                            else -> "Período de Descanso"
                        }
                        val stepDetail = if (type == "REST") "$value segundos" else if (type == "LUNGE") "$value reps (cada perna)" else "$value repetições"
                        val stepColor = if (isExercise) PrimaryCyan else Color.Gray

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = Color.White.copy(alpha = 0.03f)),
                            border = BorderStroke(1.dp, BorderMuted)
                        ) {
                            Row(
                                modifier = Modifier.padding(14.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Box(
                                        modifier = Modifier
                                            .size(24.dp)
                                            .background(stepColor.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                            .border(1.dp, stepColor.copy(alpha = 0.4f), RoundedCornerShape(6.dp)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text("${index + 1}", color = stepColor, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                    }
                                    Spacer(modifier = Modifier.width(12.dp))
                                    Column {
                                        Text(stepTitle, color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                                        Text(stepDetail, color = TextSecondary, fontSize = 11.sp)
                                    }
                                }

                                // Delete Step Button
                                IconButton(onClick = { steps.removeAt(index) }) {
                                    Text("✕", color = Color(0xFFFF4D4D), fontSize = 14.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(36.dp))

            // SAVE PLAN button
            Button(
                onClick = {
                    if (planName.isBlank()) {
                        Toast.makeText(context, context.getString(R.string.toast_missing_plan_name), Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    val exercisesOnly = steps.filter { it["type"] != "REST" }
                    if (exercisesOnly.isEmpty()) {
                        Toast.makeText(context, context.getString(R.string.toast_missing_exercise), Toast.LENGTH_SHORT).show()
                        return@Button
                    }

                    // Enforce rest constraints: no consecutive exercises without rest
                    for (i in 0 until steps.size - 1) {
                        val first = steps[i]["type"] as String
                        val second = steps[i + 1]["type"] as String
                        if (first != "REST" && second != "REST") {
                            Toast.makeText(context, context.getString(R.string.toast_missing_rest), Toast.LENGTH_LONG).show()
                            return@Button
                        }
                    }

                    // Save custom plan to Firestore
                    val uid = FirebaseAuth.getInstance().currentUser?.uid
                    if (uid != null) {
                        val db = FirebaseFirestore.getInstance()
                        val stepsJsonArray = JSONArray()
                        steps.forEach { step ->
                            val obj = JSONObject().apply {
                                put("type", step["type"] as String)
                                put("value", step["value"] as Int)
                            }
                            stepsJsonArray.put(obj)
                        }
                        val stepsJsonStr = stepsJsonArray.toString()

                        val planData = hashMapOf(
                            "planName" to planName,
                            "createdAt" to Timestamp(Date()),
                            "stepsJson" to stepsJsonStr
                        )

                        db.collection("users").document(uid).collection("custom_plans").add(planData)
                            .addOnSuccessListener {
                                Toast.makeText(context, context.getString(R.string.toast_plan_saved), Toast.LENGTH_SHORT).show()
                                onSaved(planName, stepsJsonStr)
                            }
                            .addOnFailureListener { e ->
                                Toast.makeText(context, context.getString(R.string.toast_save_error, e.localizedMessage), Toast.LENGTH_SHORT).show()
                            }
                    } else {
                        // Offline bypass for testing
                        val stepsJsonArray = JSONArray()
                        steps.forEach { step ->
                            val obj = JSONObject().apply {
                                put("type", step["type"] as String)
                                put("value", step["value"] as Int)
                            }
                            stepsJsonArray.put(obj)
                        }
                        onSaved(planName, stepsJsonArray.toString())
                    }
                },
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
                        text = "GUARDAR E PREPARAR TREINO",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0C0F14),
                        letterSpacing = 1.2.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(48.dp))
        }
    }
}
