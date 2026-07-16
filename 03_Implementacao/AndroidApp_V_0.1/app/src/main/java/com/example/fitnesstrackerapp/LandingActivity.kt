package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*
import com.google.firebase.auth.FirebaseAuth

import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat

class LandingActivity : AppCompatActivity() {
    private lateinit var auth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        
        auth = FirebaseAuth.getInstance()
        
        // Auto-login if session exists
        if (auth.currentUser != null) {
            navigateToDashboard()
            return
        }

        enableEdgeToEdge()
        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    LandingScreen(
                        onLoginSuccess = { navigateToDashboard() },
                        onNavigateToRegister = {
                            val intent = Intent(this@LandingActivity, RegisterActivity::class.java)
                            startActivity(intent)
                        }
                    )
                }
            }
        }
    }

    private fun navigateToDashboard() {
        val intent = Intent(this, LoaderActivity::class.java)
        intent.putExtra("TARGET_ACTIVITY", DashboardActivity::class.java.name)
        startActivity(intent)
        finish()
    }
}

@Composable
fun LandingScreen(
    onLoginSuccess: () -> Unit,
    onNavigateToRegister: () -> Unit
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showResetDialog by remember { mutableStateOf(false) }

    if (showResetDialog) {
        var resetEmail by remember { mutableStateOf("") }
        var isResetting by remember { mutableStateOf(false) }
        var resetSuccessMessage by remember { mutableStateOf<String?>(null) }
        var resetErrorMessage by remember { mutableStateOf<String?>(null) }

        AlertDialog(
            onDismissRequest = { showResetDialog = false },
            title = { Text(stringResource(R.string.reset_password_title), color = Color.White) },
            containerColor = DarkSurface,
            text = {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = stringResource(R.string.reset_password_desc),
                        color = TextSecondary,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                    OutlinedTextField(
                        value = resetEmail,
                        onValueChange = { resetEmail = it },
                        label = { Text(stringResource(R.string.email_label), color = TextSecondary) },
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = PrimaryCyan,
                            unfocusedBorderColor = BorderMuted
                        ),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        singleLine = true
                    )
                    if (resetSuccessMessage != null) {
                        Text(
                            text = resetSuccessMessage!!,
                            color = AccentGreen,
                            fontSize = 13.sp,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                    if (resetErrorMessage != null) {
                        Text(
                            text = resetErrorMessage!!,
                            color = Color.Red,
                            fontSize = 13.sp,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (resetEmail.isBlank()) {
                            resetErrorMessage = "Please enter your email."
                            return@Button
                        }
                        isResetting = true
                        resetErrorMessage = null
                        resetSuccessMessage = null
                        FirebaseAuth.getInstance().sendPasswordResetEmail(resetEmail.trim())
                            .addOnCompleteListener { task ->
                                isResetting = false
                                if (task.isSuccessful) {
                                    resetSuccessMessage = "Password reset email sent!"
                                } else {
                                    resetErrorMessage = task.exception?.localizedMessage ?: "Failed to send reset email."
                                }
                            }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan),
                    enabled = !isResetting
                ) {
                    if (isResetting) {
                        CircularProgressIndicator(color = Color(0xFF0C0F14), modifier = Modifier.size(16.dp))
                    } else {
                        Text(stringResource(R.string.btn_send), color = Color(0xFF0C0F14), fontWeight = FontWeight.Bold)
                    }
                }
            },
            dismissButton = {
                TextButton(
                    onClick = { showResetDialog = false },
                    enabled = !isResetting
                ) {
                    Text(stringResource(R.string.btn_close), color = TextSecondary)
                }
            }
        )
    }

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
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Spacer(modifier = Modifier.height(24.dp))

            // App Title / Branding
            Text(
                text = "FITNESS",
                fontSize = 44.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryCyan,
                letterSpacing = 4.sp,
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(6.dp))
            
            Text(
                text = "TRACKER",
                fontSize = 44.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryCyan,
                letterSpacing = 4.sp,
                textAlign = TextAlign.Center
            )
            
            Text(
                text = "Train anywhere you want!",
                fontSize = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextSecondary,
                letterSpacing = 2.sp,
                modifier = Modifier.padding(top = 16.dp, bottom = 40.dp),
                textAlign = TextAlign.Center
            )

        // Login Box
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = DarkSurface),
            border = androidx.compose.foundation.BorderStroke(1.dp, BorderMuted)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = stringResource(R.string.login_title),
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.Start)
                        .padding(bottom = 16.dp)
                )

                OutlinedTextField(
                    value = email,
                    onValueChange = { 
                        email = it
                        errorMessage = null 
                    },
                    label = { Text(stringResource(R.string.email_label), color = TextSecondary) },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = BorderMuted,
                        cursorColor = PrimaryCyan
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true,
                    enabled = !isLoading
                )

                OutlinedTextField(
                    value = password,
                    onValueChange = { 
                        password = it
                        errorMessage = null
                    },
                    label = { Text(stringResource(R.string.password_label), color = TextSecondary) },
                    visualTransformation = PasswordVisualTransformation(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = Color.White,
                        unfocusedTextColor = Color.White,
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = BorderMuted,
                        cursorColor = PrimaryCyan
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp),
                    shape = RoundedCornerShape(12.dp),
                    singleLine = true,
                    enabled = !isLoading
                )

                Text(
                    text = stringResource(R.string.forgot_password),
                    color = PrimaryCyan,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier
                        .align(Alignment.End)
                        .clickable { showResetDialog = true }
                        .padding(vertical = 4.dp)
                )

                // Error Message Display
                if (errorMessage != null) {
                    Text(
                        text = errorMessage ?: "",
                        color = Color.Red,
                        fontSize = 13.sp,
                        modifier = Modifier.padding(top = 8.dp),
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
            }
        }

        // Action Button
        Button(
            onClick = {
                if (email.isBlank() || password.isBlank()) {
                    errorMessage = "Please fill in all fields."
                    return@Button
                }
                isLoading = true
                errorMessage = null
                FirebaseAuth.getInstance().signInWithEmailAndPassword(email.trim(), password)
                    .addOnCompleteListener { task ->
                        if (task.isSuccessful) {
                            val uid = task.result?.user?.uid
                            if (uid != null) {
                                com.google.firebase.firestore.FirebaseFirestore.getInstance().collection("users").document(uid).get()
                                    .addOnSuccessListener { doc ->
                                        isLoading = false
                                        val lang = doc.getString("preferredLanguage") ?: "SYSTEM"
                                        if (lang != "SYSTEM") {
                                            AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(lang))
                                        } else {
                                            AppCompatDelegate.setApplicationLocales(LocaleListCompat.getEmptyLocaleList())
                                        }
                                        onLoginSuccess()
                                    }
                                    .addOnFailureListener {
                                        isLoading = false
                                        onLoginSuccess()
                                    }
                            } else {
                                isLoading = false
                                onLoginSuccess()
                            }
                        } else {
                            isLoading = false
                            errorMessage = task.exception?.localizedMessage ?: "Login failed. Please try again."
                        }
                    }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.Transparent
            ),
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
                        text = stringResource(R.string.btn_sign_in),
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF0C0F14),
                        letterSpacing = 1.5.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Navigation link to Register
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                text = stringResource(R.string.no_account),
                color = TextSecondary,
                fontSize = 14.sp
            )
            Text(
                text = stringResource(R.string.create_account),
                color = PrimaryCyan,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.clickable { onNavigateToRegister() }
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "v1.0.0-beta",
            fontSize = 12.sp,
            color = BorderMuted,
            textAlign = TextAlign.Center
        )
    }
}
}
