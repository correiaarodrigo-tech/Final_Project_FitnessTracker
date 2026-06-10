package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fitnesstrackerapp.ui.theme.*
import androidx.compose.runtime.*
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.example.fitnesstrackerapp.logic.UserProfile

class DashboardActivity : ComponentActivity() {
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
                    DashboardScreen(
                        onNavigate = { activityClass ->
                            val intent = if (activityClass == MainActivity::class.java || activityClass == DemoPushUpActivity::class.java) {
                                Intent(this@DashboardActivity, LoaderActivity::class.java).apply {
                                    putExtra("TARGET_ACTIVITY", activityClass.name)
                                }
                            } else {
                                Intent(this@DashboardActivity, activityClass)
                            }
                            startActivity(intent)
                        },
                        onLogout = {
                            val intent = Intent(this@DashboardActivity, LandingActivity::class.java)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                            startActivity(intent)
                            finish()
                        }
                    )
                }
            }
        }
    }
}

data class DashboardItem(
    val title: String,
    val subtitle: String,
    val badge: String?,
    val accentColor: Color,
    val targetActivity: Class<*>
)

@Composable
fun DashboardScreen(
    onNavigate: (Class<*>) -> Unit,
    onLogout: () -> Unit
) {
    val auth = remember { FirebaseAuth.getInstance() }
    val db = remember { FirebaseFirestore.getInstance() }
    val currentUser = auth.currentUser
    
    var userName by remember { mutableStateOf("Athlete") }
    var userCode by remember { mutableStateOf("") }
    
    LaunchedEffect(currentUser) {
        currentUser?.uid?.let { uid ->
            db.collection("users").document(uid).get()
                .addOnSuccessListener { doc ->
                    if (doc.exists()) {
                        val profile = doc.toObject(UserProfile::class.java)
                        if (profile != null) {
                            userName = profile.name
                            userCode = profile.numericId
                        }
                    }
                }
        }
    }

    val firstName = userName.trim().split("\\s+".toRegex()).firstOrNull() ?: userName

    val items = listOf(
        DashboardItem(
            title = "Edit Profile",
            subtitle = "Manage details, targets, and goals",
            badge = null,
            accentColor = PrimaryCyan,
            targetActivity = EditProfileActivity::class.java
        ),
        DashboardItem(
            title = "Create Plan",
            subtitle = "Build customized training routines",
            badge = "Builder",
            accentColor = SecondaryPurple,
            targetActivity = CreatePlanActivity::class.java
        ),
        DashboardItem(
            title = "Start a Plan",
            subtitle = "Select and follow a routine",
            badge = "Active",
            accentColor = AccentGreen,
            targetActivity = StartPlanActivity::class.java
        ),
        DashboardItem(
            title = "View Statistics",
            subtitle = "Track metrics and consistency over time",
            badge = "Analytics",
            accentColor = PrimaryCyan,
            targetActivity = ViewStatisticsActivity::class.java
        ),
        DashboardItem(
            title = "Demo Workout",
            subtitle = "Multi-exercise AI pose-tracking routine",
            badge = "Beta Live",
            accentColor = AccentGreen,
            targetActivity = MainActivity::class.java
        ),
        DashboardItem(
            title = "Demo Pushups",
            subtitle = "AI push-up repetition counter demo",
            badge = "Beta Live",
            accentColor = SecondaryPurple,
            targetActivity = DemoPushUpActivity::class.java
        )
    )

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
        ) {
        Spacer(modifier = Modifier.height(24.dp))
        
        // Welcome Header
        Text(
            text = if (userCode.isNotEmpty()) "Welcome Back, $firstName" else "Welcome Back",
            fontSize = 16.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )
        
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp, bottom = 24.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = if (userCode.isNotEmpty()) "Athlete $userCode" else "Athlete Dashboard",
                fontSize = 28.sp,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "Sign Out",
                color = Color.Red,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier
                    .clickable {
                        FirebaseAuth.getInstance().signOut()
                        onLogout()
                    }
                    .padding(8.dp)
            )
        }

        // Navigation Grid
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.fillMaxSize()
        ) {
            items(items) { item ->
                DashboardCard(
                    title = item.title,
                    subtitle = item.subtitle,
                    badgeText = item.badge,
                    accentColor = item.accentColor,
                    onClick = { onNavigate(item.targetActivity) }
                )
            }
        }
    }
}
}

@Composable
fun DashboardCard(
    title: String,
    subtitle: String,
    badgeText: String?,
    accentColor: Color,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(140.dp)
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = androidx.compose.foundation.BorderStroke(1.dp, BorderMuted)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Colored dot indicator
                androidx.compose.foundation.Canvas(modifier = Modifier.size(8.dp)) {
                    drawCircle(color = accentColor)
                }
                
                if (badgeText != null) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = accentColor.copy(alpha = 0.15f)),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Text(
                            text = badgeText,
                            color = accentColor,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                        )
                    }
                }
            }

            Column {
                Text(
                    text = title,
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = subtitle,
                    color = TextSecondary,
                    fontSize = 11.sp,
                    lineHeight = 14.sp,
                    maxLines = 2
                )
            }
        }
    }
}
