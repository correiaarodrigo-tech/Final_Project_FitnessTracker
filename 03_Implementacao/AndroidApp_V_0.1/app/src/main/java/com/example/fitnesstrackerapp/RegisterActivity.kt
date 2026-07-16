package com.example.fitnesstrackerapp

import android.app.DatePickerDialog
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.logic.UserProfile
import com.example.fitnesstrackerapp.ui.theme.*
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import java.text.SimpleDateFormat
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import java.util.*

class RegisterActivity : AppCompatActivity() {
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
                    RegisterScreen(
                        onRegisterSuccess = {
                            val intent = Intent(this@RegisterActivity, DashboardActivity::class.java)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                            startActivity(intent)
                        },
                        onBack = { finish() }
                    )
                }
            }
        }
    }
}

@Composable
fun RegisterScreen(
    onRegisterSuccess: () -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var weightStr by remember { mutableStateOf("") }
    var heightStr by remember { mutableStateOf("") }
    
    // Gender selection: MALE, FEMALE, OTHER
    var selectedGender by remember { mutableStateOf("OTHER") }
    
    // Language selection: SYSTEM, en, pt
    var selectedLanguage by remember { mutableStateOf("SYSTEM") }
    
    // DOB representation
    var dobTimestamp by remember { mutableStateOf(0L) }
    val dobFormatter = remember { SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()) }
    val dobDisplay = if (dobTimestamp == 0L) "Select Date of Birth" else dobFormatter.format(Date(dobTimestamp))

    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    val scrollState = rememberScrollState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 480.dp)
                .fillMaxWidth()
                .padding(24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(scrollState),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
        // Top Back Navigation
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Start
        ) {
            Text(
                text = "← Back",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryCyan,
                modifier = Modifier
                    .clickable(onClick = onBack)
                    .padding(vertical = 8.dp)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = stringResource(R.string.register_title),
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        
        Text(
            text = "Fill in your profile details to get started",
            fontSize = 14.sp,
            color = TextSecondary,
            modifier = Modifier.padding(top = 4.dp, bottom = 24.dp)
        )

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = DarkSurface),
            border = BorderStroke(1.dp, BorderMuted)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text(stringResource(R.string.name_label), color = TextSecondary) },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = BorderMuted,
                        cursorColor = PrimaryCyan
                    ),
                    modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true,
                    enabled = !isLoading
                )

                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text(stringResource(R.string.email_label), color = TextSecondary) },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = BorderMuted,
                        cursorColor = PrimaryCyan
                    ),
                    modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true,
                    enabled = !isLoading
                )

                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text(stringResource(R.string.password_label), color = TextSecondary) },
                    visualTransformation = PasswordVisualTransformation(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = BorderMuted,
                        cursorColor = PrimaryCyan
                    ),
                    modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true,
                    enabled = !isLoading
                )

                Divider(color = BorderMuted, modifier = Modifier.padding(vertical = 8.dp))

                Text(
                    text = "Biometric Details",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.Start)
                        .padding(vertical = 8.dp)
                )

                // Date of Birth Button
                OutlinedButton(
                    onClick = {
                        val calendar = Calendar.getInstance()
                        // If we already selected one, set to that
                        if (dobTimestamp != 0L) {
                            calendar.timeInMillis = dobTimestamp
                        }
                        DatePickerDialog(
                            context,
                            { _, year, month, dayOfMonth ->
                                val selectedCal = Calendar.getInstance()
                                selectedCal.set(year, month, dayOfMonth)
                                dobTimestamp = selectedCal.timeInMillis
                            },
                            calendar.get(Calendar.YEAR),
                            calendar.get(Calendar.MONTH),
                            calendar.get(Calendar.DAY_OF_MONTH)
                        ).show()
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    shape = RoundedCornerShape(12.dp),
                    border = BorderStroke(1.dp, if (dobTimestamp != 0L) PrimaryCyan else BorderMuted),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.White),
                    enabled = !isLoading
                ) {
                    Text(
                        text = dobDisplay,
                        color = if (dobTimestamp != 0L) Color.White else TextSecondary,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Start
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    OutlinedTextField(
                        value = weightStr,
                        onValueChange = { weightStr = it },
                        label = { Text("Weight (kg)", color = TextSecondary) },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = PrimaryCyan,
                            unfocusedBorderColor = BorderMuted,
                            cursorColor = PrimaryCyan
                        ),
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        singleLine = true,
                        enabled = !isLoading
                    )

                    OutlinedTextField(
                        value = heightStr,
                        onValueChange = { heightStr = it },
                        label = { Text("Height (cm)", color = TextSecondary) },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = PrimaryCyan,
                            unfocusedBorderColor = BorderMuted,
                            cursorColor = PrimaryCyan
                        ),
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(12.dp),
                        singleLine = true,
                        enabled = !isLoading
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Language selector row
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Preferred Language",
                        fontSize = 14.sp,
                        color = TextSecondary,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf("SYSTEM" to "System", "en" to "EN", "pt" to "PT").forEach { (code, label) ->
                            val isSelected = selectedLanguage == code
                            Button(
                                onClick = { selectedLanguage = code },
                                modifier = Modifier.weight(1f),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isSelected) PrimaryCyan else ComponentBg,
                                    contentColor = if (isSelected) Color(0xFF0C0F14) else Color.White
                                ),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Text(label, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                // Gender selector row
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = stringResource(R.string.sign_in_link),
                        fontSize = 14.sp,
                        color = TextSecondary,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf("MALE", "FEMALE", "OTHER").forEach { gender ->
                            val isSelected = selectedGender == gender
                            val label = when (gender) {
                                "MALE" -> "Male"
                                "FEMALE" -> "Female"
                                else -> "Prefer not to say"
                            }
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .background(
                                        color = if (isSelected) PrimaryCyan.copy(alpha = 0.15f) else Color.Transparent,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .border(
                                        width = 1.dp,
                                        color = if (isSelected) PrimaryCyan else BorderMuted,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .clickable(enabled = !isLoading) { selectedGender = gender }
                                    .padding(vertical = 12.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = label,
                                    color = if (isSelected) PrimaryCyan else TextSecondary,
                                    fontSize = 12.sp,
                                    fontWeight = FontWeight.Bold,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                }

                if (errorMessage != null) {
                    Text(
                        text = errorMessage ?: "",
                        color = Color.Red,
                        fontSize = 13.sp,
                        modifier = Modifier.padding(top = 16.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }

        // Register Action Button
        Button(
            onClick = {
                val weight = weightStr.toFloatOrNull()
                val height = heightStr.toFloatOrNull()

                if (name.isBlank() || email.isBlank() || password.isBlank() || dobTimestamp == 0L || weight == null || height == null) {
                    errorMessage = "Please enter valid profile details."
                    return@Button
                }

                if (password.length < 6) {
                    errorMessage = "Password must be at least 6 characters."
                    return@Button
                }

                isLoading = true
                errorMessage = null

                val mAuth = FirebaseAuth.getInstance()
                mAuth.createUserWithEmailAndPassword(email.trim(), password)
                    .addOnCompleteListener { task ->
                        if (task.isSuccessful) {
                            val uid = task.result.user?.uid ?: ""
                            val randomNum = (10000..99999).random()
                            val numericId = "#$randomNum"
                            val profile = UserProfile(
                                uid = uid,
                                name = name,
                                numericId = numericId,
                                dob = com.google.firebase.Timestamp(java.util.Date(dobTimestamp)),
                                weightKg = weight,
                                heightCm = height,
                                gender = selectedGender,
                                preferredLanguage = selectedLanguage,
                                xpPoints = 0,
                                level = 1,
                                friendsList = emptyList(),
                                createdAt = com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis())),
                                modifiedAt = com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis())),
                                lastActive = com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis()))
                            )

                            // Save to Firestore
                            val db = FirebaseFirestore.getInstance()
                            db.collection("users").document(uid).set(profile)
                                .addOnSuccessListener {
                                    FirebaseAuth.getInstance().signOut()
                                    
                                    // Apply selected language
                                    if (selectedLanguage != "SYSTEM") {
                                        AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(selectedLanguage))
                                    } else {
                                        AppCompatDelegate.setApplicationLocales(LocaleListCompat.getEmptyLocaleList())
                                    }
                                    
                                    isLoading = false
                                    onRegisterSuccess()
                                }
                                .addOnFailureListener { e ->
                                    isLoading = false
                                    errorMessage = "Saved account but profile sync failed: ${e.localizedMessage}"
                                }
                        } else {
                            isLoading = false
                            errorMessage = task.exception?.localizedMessage ?: "Registration failed."
                        }
                    }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
            contentPadding = PaddingValues(),
            enabled = !isLoading
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
                if (isLoading) {
                    CircularProgressIndicator(color = Color(0xFF0C0F14), modifier = Modifier.size(24.dp))
                } else {
                    Text(
                        text = stringResource(R.string.btn_register),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0C0F14),
                        letterSpacing = 1.5.sp
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(24.dp))
    }
}
}
