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
import androidx.compose.ui.res.stringResource
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
                        onNavigate = { activityClass, exerciseType ->
                            val isCameraActivity = activityClass == MainActivity::class.java ||
                                activityClass == DemoPushUpActivity::class.java ||
                                activityClass == ExerciseTestActivity::class.java
                            val intent = if (isCameraActivity) {
                                Intent(this@DashboardActivity, LoaderActivity::class.java).apply {
                                    putExtra("TARGET_ACTIVITY", activityClass.name)
                                    if (exerciseType != null) putExtra("EXERCISE_TYPE", exerciseType)
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
    val targetActivity: Class<*>,
    val exerciseType: String? = null
)

@Composable
fun DashboardScreen(
    onNavigate: (Class<*>, String?) -> Unit,
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
    var pendingRequests by remember { mutableStateOf<List<Map<String, Any>>>(emptyList()) }
    
    DisposableEffect(currentUser) {
        var profileListener: com.google.firebase.firestore.ListenerRegistration? = null
        currentUser?.uid?.let { uid ->
            db.collection("users").document(uid)
                .update("lastActive", com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis())))
                .addOnFailureListener { e ->
                    android.util.Log.w("DashboardActivity", "Failed to update lastActive", e)
                }
            
            profileListener = db.collection("users").document(uid)
                .addSnapshotListener { snapshot, error ->
                    if (error != null) {
                        android.util.Log.e("DashboardActivity", "Error listening to user profile", error)
                        Toast.makeText(context, "Error loading profile: ${error.localizedMessage}", Toast.LENGTH_SHORT).show()
                        return@addSnapshotListener
                    }
                    if (snapshot != null && snapshot.exists()) {
                        try {
                            val profile = snapshot.toObject(UserProfile::class.java)
                            if (profile != null) {
                                userName = profile.name
                                userCode = profile.numericId
                                friendsCodes = profile.friendsList
                                android.util.Log.d("DashboardActivity", "User profile loaded: name=${profile.name}, code=${profile.numericId}, friends=${profile.friendsList}")
                            }
                        } catch (e: Exception) {
                            android.util.Log.e("DashboardActivity", "Error parsing UserProfile", e)
                        }
                    }
                }
        }
        onDispose {
            profileListener?.remove()
        }
    }

    DisposableEffect(friendsCodes) {
        var friendsListener: com.google.firebase.firestore.ListenerRegistration? = null
        if (friendsCodes.isEmpty()) {
            friendsProfiles = emptyList()
        } else {
            isFetchingFriends = true
            friendsListener = db.collection("users").whereIn("numericId", friendsCodes)
                .addSnapshotListener { querySnapshot, error ->
                    isFetchingFriends = false
                    if (error != null) {
                        android.util.Log.e("DashboardActivity", "Error listening to friends profiles", error)
                        Toast.makeText(context, "Error loading friends status: ${error.localizedMessage}", Toast.LENGTH_SHORT).show()
                        return@addSnapshotListener
                    }
                    if (querySnapshot != null) {
                        try {
                            val profiles = querySnapshot.documents.mapNotNull { it.toObject(UserProfile::class.java) }
                            android.util.Log.d("DashboardActivity", "Loaded ${profiles.size} friend profiles: ${profiles.map { it.name }}")
                            friendsProfiles = profiles
                        } catch (e: Exception) {
                            android.util.Log.e("DashboardActivity", "Error parsing friends profiles", e)
                        }
                    }
                }
        }
        onDispose {
            friendsListener?.remove()
        }
    }

    DisposableEffect(currentUser) {
        var requestsListener: com.google.firebase.firestore.ListenerRegistration? = null
        currentUser?.uid?.let { uid ->
            requestsListener = db.collection("friend_requests")
                .whereEqualTo("receiverUid", uid)
                .whereEqualTo("status", "PENDING")
                .addSnapshotListener { snapshot, error ->
                    if (error != null) {
                        android.util.Log.e("DashboardActivity", "Error listening to friend requests", error)
                        return@addSnapshotListener
                    }
                    if (snapshot != null) {
                        pendingRequests = snapshot.documents.mapNotNull { doc ->
                            doc.data?.toMutableMap()?.apply {
                                this["requestId"] = doc.id
                            }
                        }
                        android.util.Log.d("DashboardActivity", "Loaded ${pendingRequests.size} pending friend requests")
                    }
                }
        }
        onDispose {
            requestsListener?.remove()
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
            title = stringResource(R.string.start_plan),
            subtitle = stringResource(R.string.select_routine),
            badge = stringResource(R.string.active_badge),
            accentColor = AccentGreen,
            targetActivity = StartPlanActivity::class.java
        ),
        DashboardItem(
            title = stringResource(R.string.view_stats),
            subtitle = stringResource(R.string.track_metrics),
            badge = stringResource(R.string.analytics_badge),
            accentColor = PrimaryCyan,
            targetActivity = ViewStatisticsActivity::class.java
        ),
        DashboardItem(
            title = stringResource(R.string.demo_workout),
            subtitle = stringResource(R.string.multi_exercise),
            badge = stringResource(R.string.beta_live),
            accentColor = AccentGreen,
            targetActivity = MainActivity::class.java
        ),
        DashboardItem(
            title = stringResource(R.string.demo_pushups),
            subtitle = stringResource(R.string.ai_pushup),
            badge = stringResource(R.string.beta_live),
            accentColor = SecondaryPurple,
            targetActivity = DemoPushUpActivity::class.java
        )
    )

    // Single-exercise free practice screens (live form evaluation).
    val exerciseItems = listOf(
        DashboardItem(
            title = stringResource(R.string.push_up),
            subtitle = stringResource(R.string.track_reps_elbow),
            badge = stringResource(R.string.test_badge),
            accentColor = PrimaryCyan,
            targetActivity = ExerciseTestActivity::class.java,
            exerciseType = ExerciseType.PUSHUP
        ),
        DashboardItem(
            title = stringResource(R.string.squat),
            subtitle = stringResource(R.string.track_reps_knee),
            badge = stringResource(R.string.test_badge),
            accentColor = AccentGreen,
            targetActivity = ExerciseTestActivity::class.java,
            exerciseType = ExerciseType.SQUAT
        ),
        DashboardItem(
            title = stringResource(R.string.lunge),
            subtitle = stringResource(R.string.lunge_desc),
            badge = stringResource(R.string.test_badge),
            accentColor = SecondaryPurple,
            targetActivity = ExerciseTestActivity::class.java,
            exerciseType = ExerciseType.LUNGE
        ),
        DashboardItem(
            title = stringResource(R.string.plank),
            subtitle = stringResource(R.string.plank_desc),
            badge = stringResource(R.string.test_badge),
            accentColor = PrimaryCyan,
            targetActivity = ExerciseTestActivity::class.java,
            exerciseType = ExerciseType.PLANK
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
                text = stringResource(R.string.welcome_back, firstName),
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
                    text = if (userCode.isNotEmpty()) stringResource(R.string.athlete_prefix, userCode) else stringResource(R.string.athlete_dashboard_title),
                    fontSize = 28.sp,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = stringResource(R.string.sign_out_btn),
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
                                    onClick = { onNavigate(item.targetActivity, item.exerciseType) }
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

            // Test Exercises Section
            Text(
                text = stringResource(R.string.test_exercises),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Text(
                text = stringResource(R.string.practice_single),
                fontSize = 13.sp,
                color = TextSecondary,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                exerciseItems.chunked(2).forEach { rowItems ->
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
                                    onClick = { onNavigate(item.targetActivity, item.exerciseType) }
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
                text = stringResource(R.string.friends_status),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.padding(bottom = 12.dp)
            )
            
            // 1. Pending Requests Sub-Section
            if (pendingRequests.isNotEmpty()) {
                Text(
                    text = stringResource(R.string.pending_requests_count, pendingRequests.size),
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold,
                    color = SecondaryPurple,
                    modifier = Modifier.padding(bottom = 12.dp)
                )
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    pendingRequests.forEach { request ->
                        val senderName = request["senderName"] as? String ?: stringResource(R.string.unknown_athlete)
                        val senderCode = request["senderCode"] as? String ?: ""
                        val requestId = request["requestId"] as? String ?: ""
                        
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = CardDefaults.cardColors(containerColor = DarkSurface),
                            border = BorderStroke(1.dp, BorderMuted)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = senderName,
                                        color = Color.White,
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = stringResource(R.string.code_display, senderCode),
                                        color = TextSecondary,
                                        fontSize = 12.sp
                                    )
                                }
                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Button(
                                        onClick = {
                                            val senderUid = request["senderUid"] as? String ?: ""
                                            val receiverUid = currentUser?.uid ?: ""
                                            val receiverCode = userCode
                                            
                                            if (senderUid.isNotEmpty() && senderCode.isNotEmpty() && receiverUid.isNotEmpty() && receiverCode.isNotEmpty()) {
                                                val receiverRef = db.collection("users").document(receiverUid)
                                                db.runTransaction { transaction ->
                                                    val receiverDoc = transaction.get(receiverRef)
                                                    val senderRef = db.collection("users").document(senderUid)
                                                    val senderDoc = transaction.get(senderRef)
                                                    
                                                    val receiverFriends = (receiverDoc.get("friendsList") as? List<*>)?.mapNotNull { it as? String }?.toMutableList() ?: mutableListOf()
                                                    val senderFriends = (senderDoc.get("friendsList") as? List<*>)?.mapNotNull { it as? String }?.toMutableList() ?: mutableListOf()
                                                    
                                                    if (!receiverFriends.contains(senderCode)) receiverFriends.add(senderCode)
                                                    if (!senderFriends.contains(receiverCode)) senderFriends.add(receiverCode)
                                                    
                                                    transaction.update(receiverRef, "friendsList", receiverFriends)
                                                    transaction.update(senderRef, "friendsList", senderFriends)
                                                    transaction.delete(db.collection("friend_requests").document(requestId))
                                                }.addOnSuccessListener {
                                                    Toast.makeText(context, "Friend request accepted!", Toast.LENGTH_SHORT).show()
                                                }.addOnFailureListener { e ->
                                                    Toast.makeText(context, "Failed to accept: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                }
                                            }
                                        },
                                        colors = ButtonDefaults.buttonColors(containerColor = AccentGreen),
                                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                                        shape = RoundedCornerShape(8.dp)
                                    ) {
                                        Text(stringResource(R.string.btn_accept), color = Color(0xFF0C0F14), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                    }
                                    
                                    Button(
                                        onClick = {
                                            if (requestId.isNotEmpty()) {
                                                db.collection("friend_requests").document(requestId).delete()
                                                    .addOnSuccessListener {
                                                        Toast.makeText(context, "Friend request declined.", Toast.LENGTH_SHORT).show()
                                                    }
                                            }
                                        },
                                        colors = ButtonDefaults.buttonColors(containerColor = Color.Red),
                                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                                        shape = RoundedCornerShape(8.dp)
                                    ) {
                                        Text(stringResource(R.string.btn_decline), color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                    }
                                }
                            }
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(24.dp))
            }
            
            // 2. Friends Online/Offline status section
            if (isFetchingFriends) {
                Box(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = PrimaryCyan)
                }
            } else if (friendsCodes.isEmpty()) {
                Text(
                    text = stringResource(R.string.no_friends_yet),
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
    val statusText = run {
        val lastActiveMs = friend.getLastActiveLong()
        val now = System.currentTimeMillis()
        val diff = now - lastActiveMs
        when {
            diff < 5 * 60 * 1000 -> stringResource(R.string.status_online)
            diff < 60 * 60 * 1000 -> {
                val mins = diff / (60 * 1000)
                stringResource(R.string.status_mins_ago, mins.toInt())
            }
            diff < 24 * 60 * 60 * 1000 -> {
                val hours = diff / (60 * 60 * 1000)
                stringResource(R.string.status_hours_ago, hours.toInt())
            }
            diff < 72 * 60 * 60 * 1000 -> {
                val days = diff / (24 * 60 * 60 * 1000)
                stringResource(R.string.status_days_ago, days.toInt())
            }
            else -> stringResource(R.string.status_old)
        }
    }
    
    val isOnline = statusText == stringResource(R.string.status_online)

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
                    Text(stringResource(R.string.btn_challenge), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
                
                // Stats Button
                TextButton(
                    onClick = onViewStats,
                    colors = ButtonDefaults.textButtonColors(contentColor = SecondaryPurple),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(stringResource(R.string.btn_stats), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
