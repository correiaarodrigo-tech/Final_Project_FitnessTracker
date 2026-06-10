package com.example.fitnesstrackerapp

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
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
    val context = LocalContext.current
    
    var userName by remember { mutableStateOf("Athlete") }
    var userCode by remember { mutableStateOf("") }
    var friendsCodes by remember { mutableStateOf<List<String>>(emptyList()) }
    var friendsProfiles by remember { mutableStateOf<List<UserProfile>>(emptyList()) }
    var isFetchingFriends by remember { mutableStateOf(false) }
    
    LaunchedEffect(currentUser) {
        currentUser?.uid?.let { uid ->
            // Update lastActive timestamp in Firestore
            db.collection("users").document(uid)
                .update("lastActive", com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis())))
                .addOnFailureListener {
                    // Fallback if update fails
                }
            
            db.collection("users").document(uid).get()
                .addOnSuccessListener { doc ->
                    if (doc.exists()) {
                        val profile = doc.toObject(UserProfile::class.java)
                        if (profile != null) {
                            userName = profile.name
                            userCode = profile.numericId
                            friendsCodes = profile.friendsList
                        }
                    }
                }
        }
    }

    LaunchedEffect(friendsCodes) {
        if (friendsCodes.isEmpty()) {
            friendsProfiles = emptyList()
            return@LaunchedEffect
        }
        isFetchingFriends = true
        db.collection("users").whereIn("numericId", friendsCodes).get()
            .addOnSuccessListener { querySnapshot ->
                isFetchingFriends = false
                val list = querySnapshot.documents.mapNotNull { it.toObject(UserProfile::class.java) }
                friendsProfiles = list
            }
            .addOnFailureListener {
                isFetchingFriends = false
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
                .verticalScroll(rememberScrollState())
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

            // Navigation Grid (replaced LazyVerticalGrid with Row chunk columns for scrollable parent compatibility)
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                val chunkedItems = items.chunked(2)
                chunkedItems.forEach { rowItems ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        rowItems.forEach { item ->
                            Box(modifier = Modifier.weight(1f)) {
                                DashboardCard(
                                    title = item.title,
                                    subtitle = item.subtitle,
                                    badgeText = item.badge,
                                    accentColor = item.accentColor,
                                    onClick = { onNavigate(item.targetActivity) }
                                )
                            }
                        }
                        if (rowItems.size < 2) {
                            Spacer(modifier = Modifier.weight(1f))
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // Friends List Header
            Text(
                text = "Friends Status",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            if (isFetchingFriends) {
                Box(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = PrimaryCyan)
                }
            } else if (friendsCodes.isEmpty()) {
                Text(
                    text = "No friends added yet. Go to Edit Profile to add friends by their code!",
                    color = TextSecondary,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(vertical = 12.dp)
                )
            } else {
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    friendsProfiles.forEach { friend ->
                        FriendListItem(
                            friend = friend,
                            onChallenge = {
                                Toast.makeText(context, "Challenge issued to ${friend.name}!", Toast.LENGTH_SHORT).show()
                            },
                            onViewStats = {
                                Toast.makeText(context, "Viewing stats for ${friend.name}...", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(32.dp))
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

@Composable
fun FriendListItem(
    friend: UserProfile,
    onChallenge: () -> Unit,
    onViewStats: () -> Unit
) {
    val statusText = remember(friend) {
        val lastActiveMs = friend.getLastActiveLong()
        val now = System.currentTimeMillis()
        val diff = now - lastActiveMs
        when {
            diff < 5 * 60 * 1000 -> "Online"
            diff < 60 * 60 * 1000 -> {
                val mins = diff / (60 * 1000)
                "Active ${mins}m ago"
            }
            diff < 24 * 60 * 60 * 1000 -> {
                val hours = diff / (60 * 60 * 1000)
                "Active ${hours}h ago"
            }
            diff < 72 * 60 * 60 * 1000 -> {
                val days = diff / (24 * 60 * 60 * 1000)
                "Active ${days}d ago"
            }
            else -> "Active >72h ago"
        }
    }
    
    val isOnline = statusText == "Online"

    Card(
        modifier = Modifier
            .fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = DarkSurface),
        border = BorderStroke(1.dp, BorderMuted)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = friend.name,
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Status dot
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(
                                color = if (isOnline) AccentGreen else BorderMuted,
                                shape = RoundedCornerShape(4.dp)
                            )
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "${friend.numericId} • $statusText",
                        color = if (isOnline) AccentGreen else TextSecondary,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
            
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // Challenge Button
                TextButton(
                    onClick = onChallenge,
                    colors = ButtonDefaults.textButtonColors(contentColor = PrimaryCyan),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text("Challenge", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
                
                // Stats Button
                TextButton(
                    onClick = onViewStats,
                    colors = ButtonDefaults.textButtonColors(contentColor = SecondaryPurple),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text("Stats", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
