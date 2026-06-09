package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
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

class DashboardActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    DashboardScreen(
                        onNavigate = { activityClass ->
                            val intent = Intent(this@DashboardActivity, activityClass)
                            startActivity(intent)
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
fun DashboardScreen(onNavigate: (Class<*>) -> Unit) {
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

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp)
            .statusBarsPadding()
            .navigationBarsPadding()
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        
        // Welcome Header
        Text(
            text = "Welcome Back",
            fontSize = 16.sp,
            color = TextSecondary,
            fontWeight = FontWeight.Medium
        )
        
        Text(
            text = "Athlete Dashboard",
            fontSize = 28.sp,
            color = Color.White,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 4.dp, bottom = 24.dp)
        )

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
