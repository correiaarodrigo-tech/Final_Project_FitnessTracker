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
            
            // 1. Pending Requests Sub-Section
            if (pendingRequests.isNotEmpty()) {
                Text(
                    text = "Pending Requests (${pendingRequests.size})",
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
                        val senderName = request["senderName"] as? String ?: "Unknown Athlete"
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
                                        text = "Code: $senderCode",
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
                                        Text("Accept", color = Color(0xFF0C0F14), fontSize = 12.sp, fontWeight = FontWeight.Bold)
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
                                        Text("Decline", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
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
