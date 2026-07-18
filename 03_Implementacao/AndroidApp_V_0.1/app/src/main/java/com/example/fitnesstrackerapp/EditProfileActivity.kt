package com.example.fitnesstrackerapp

import android.app.DatePickerDialog
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
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

class EditProfileActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        
        val uid = FirebaseAuth.getInstance().currentUser?.uid
        if (uid == null) {
            Toast.makeText(this, "User not authenticated!", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        enableEdgeToEdge()
        setContent {
            FitnessTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    EditProfileScreen(
                        uid = uid,
                        onBack = { finish() }
                    )
                }
            }
        }
    }
}

@Composable
fun EditProfileScreen(
    uid: String,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val db = remember { FirebaseFirestore.getInstance() }

    // Form states
    var name by remember { mutableStateOf("") }
    var numericId by remember { mutableStateOf("") }
    var dobTimestamp by remember { mutableStateOf(0L) }
    var weightStr by remember { mutableStateOf("") }
    var heightStr by remember { mutableStateOf("") }
    var selectedGender by remember { mutableStateOf("OTHER") }
    var selectedLanguage by remember { mutableStateOf("SYSTEM") }
    var friendsList by remember { mutableStateOf<List<String>>(emptyList()) }
    var friendsNamesMap by remember { mutableStateOf<Map<String, String>>(emptyMap()) }
    var xpPoints by remember { mutableStateOf(0) }
    var level by remember { mutableStateOf(1) }
    var createdAt by remember { mutableStateOf(0L) }

    // App state
    var isFetching by remember { mutableStateOf(true) }
    var isSaving by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    // Add friend state
    var friendUidInput by remember { mutableStateOf("") }

    val dobFormatter = remember { SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()) }
    val dobDisplay = if (dobTimestamp == 0L) "Select Date of Birth" else dobFormatter.format(Date(dobTimestamp))

    // Fetch User Profile from Firestore
    LaunchedEffect(uid) {
        db.collection("users").document(uid).get()
            .addOnSuccessListener { document ->
                isFetching = false
                if (document.exists()) {
                    val profile = document.toObject(UserProfile::class.java)
                    if (profile != null) {
                        name = profile.name
                        numericId = profile.numericId
                        dobTimestamp = profile.getDobLong()
                        weightStr = profile.weightKg.toString()
                        heightStr = profile.heightCm.toString()
                        selectedGender = profile.gender
                        selectedLanguage = profile.preferredLanguage
                        friendsList = profile.friendsList
                        xpPoints = profile.xpPoints
                        level = profile.level
                        createdAt = profile.getCreatedAtLong()
                    }
                } else {
                    errorMessage = "Profile document not found."
                }
            }
            .addOnFailureListener { e ->
                isFetching = false
                errorMessage = "Failed to load profile: ${e.localizedMessage}"
            }
    }

    LaunchedEffect(friendsList) {
        if (friendsList.isEmpty()) {
            friendsNamesMap = emptyMap()
            return@LaunchedEffect
        }
        db.collection("users").whereIn("numericId", friendsList).get()
            .addOnSuccessListener { querySnapshot ->
                val newMap = querySnapshot.documents.mapNotNull { 
                    val p = it.toObject(UserProfile::class.java)
                    if (p != null) p.numericId to p.name else null
                }.toMap()
                friendsNamesMap = newMap
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
                .navigationBarsPadding(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
        // Top Navigation Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "← Back",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                color = PrimaryCyan,
                modifier = Modifier
                    .clickable(onClick = onBack)
                    .padding(8.dp)
            )
            Spacer(modifier = Modifier.weight(1f))
            Text(
                text = "Profile Settings",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Spacer(modifier = Modifier.weight(1f))
            Spacer(modifier = Modifier.width(48.dp)) // balancing back button width
        }

        if (isFetching) {
            Box(
                modifier = Modifier.weight(1f),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = PrimaryCyan)
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                // Biometrics Card
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = DarkSurface),
                        border = BorderStroke(1.dp, BorderMuted)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(20.dp)
                        ) {
                            Text(
                                text = "Personal Info",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                modifier = Modifier.padding(bottom = 12.dp)
                            )

                            OutlinedTextField(
                                value = name,
                                onValueChange = { name = it },
                                label = { Text(stringResource(R.string.full_name_label), color = TextSecondary) },
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
                                enabled = !isSaving
                            )

                            // Date of Birth Selection
                            OutlinedButton(
                                onClick = {
                                    val calendar = Calendar.getInstance()
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
                                enabled = !isSaving
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
                                    label = { Text(stringResource(R.string.weight_label), color = TextSecondary) },
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
                                    enabled = !isSaving
                                )

                                OutlinedTextField(
                                    value = heightStr,
                                    onValueChange = { heightStr = it },
                                    label = { Text(stringResource(R.string.height_label), color = TextSecondary) },
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
                                    enabled = !isSaving
                                )
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            // Language selector row
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
                                            containerColor = if (isSelected) PrimaryCyan else Color.Transparent,
                                            contentColor = if (isSelected) Color(0xFF0C0F14) else Color.White
                                        ),
                                        shape = RoundedCornerShape(12.dp)
                                    ) {
                                        Text(label, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            // Gender selection row
                            Text(
                                text = "Gender",
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
                                        else -> "Prefer not"
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
                                            .clickable(enabled = !isSaving) { selectedGender = gender }
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
                    }
                }

                // Friends List Card
                item {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(20.dp),
                        colors = CardDefaults.cardColors(containerColor = DarkSurface),
                        border = BorderStroke(1.dp, BorderMuted)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(20.dp)
                        ) {
                            Text(
                                text = "Friends List (${friendsList.size})",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.White,
                                modifier = Modifier.padding(bottom = 12.dp)
                            )

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                OutlinedTextField(
                                    value = friendUidInput,
                                    onValueChange = { friendUidInput = it },
                                    label = { Text(stringResource(R.string.add_friend_hint), color = TextSecondary) },
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedTextColor = Color.White,
                                        unfocusedTextColor = Color.White,
                                        focusedBorderColor = PrimaryCyan,
                                        unfocusedBorderColor = BorderMuted
                                    ),
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(12.dp),
                                    singleLine = true,
                                    enabled = !isSaving
                                )
                                Button(
                                    onClick = {
                                        val targetCode = friendUidInput.trim()
                                        if (targetCode == numericId) {
                                            Toast.makeText(context, "Cannot add yourself as a friend!", Toast.LENGTH_SHORT).show()
                                            return@Button
                                        }
                                        if (friendsList.contains(targetCode)) {
                                            Toast.makeText(context, "Already in your friends list!", Toast.LENGTH_SHORT).show()
                                            return@Button
                                        }
                                        if (targetCode.isNotBlank()) {
                                            isSaving = true
                                            db.collection("users").whereEqualTo("numericId", targetCode).get()
                                                .addOnSuccessListener { querySnapshot ->
                                                    if (!querySnapshot.isEmpty) {
                                                        val friendDoc = querySnapshot.documents.first()
                                                        val friendUid = friendDoc.id
                                                        val friendName = friendDoc.getString("name") ?: "Unknown"
                                                        
                                                        // Check pending requests from current user to target
                                                        db.collection("friend_requests")
                                                            .whereEqualTo("senderUid", uid)
                                                            .whereEqualTo("receiverCode", targetCode)
                                                            .whereEqualTo("status", "PENDING")
                                                            .get()
                                                            .addOnSuccessListener { reqSnapshot1 ->
                                                                if (!reqSnapshot1.isEmpty) {
                                                                    isSaving = false
                                                                    Toast.makeText(context, "Friend request already pending!", Toast.LENGTH_SHORT).show()
                                                                } else {
                                                                    // Check pending requests from target to current user
                                                                    db.collection("friend_requests")
                                                                        .whereEqualTo("senderCode", targetCode)
                                                                        .whereEqualTo("receiverUid", uid)
                                                                        .whereEqualTo("status", "PENDING")
                                                                        .get()
                                                                        .addOnSuccessListener { reqSnapshot2 ->
                                                                            if (!reqSnapshot2.isEmpty) {
                                                                                isSaving = false
                                                                                Toast.makeText(context, "This user already sent you a request! Accept it on your dashboard.", Toast.LENGTH_LONG).show()
                                                                            } else {
                                                                                // Create friend request
                                                                                val newRequest = hashMapOf(
                                                                                    "senderUid" to uid,
                                                                                    "senderCode" to numericId,
                                                                                    "senderName" to name,
                                                                                    "receiverUid" to friendUid,
                                                                                    "receiverCode" to targetCode,
                                                                                    "receiverName" to friendName,
                                                                                    "status" to "PENDING",
                                                                                    "timestamp" to com.google.firebase.Timestamp(java.util.Date())
                                                                                )
                                                                                db.collection("friend_requests").add(newRequest)
                                                                                    .addOnSuccessListener {
                                                                                        isSaving = false
                                                                                        friendUidInput = ""
                                                                                        Toast.makeText(context, "Friend request sent!", Toast.LENGTH_SHORT).show()
                                                                                    }
                                                                                    .addOnFailureListener { e ->
                                                                                        isSaving = false
                                                                                        Toast.makeText(context, "Failed to send request: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                                                    }
                                                                            }
                                                                        }
                                                                        .addOnFailureListener { e ->
                                                                            isSaving = false
                                                                            Toast.makeText(context, "Error checking inbound requests: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                                        }
                                                                }
                                                            }
                                                            .addOnFailureListener { e ->
                                                                isSaving = false
                                                                Toast.makeText(context, "Error checking outbound requests: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                            }
                                                    } else {
                                                        isSaving = false
                                                        Toast.makeText(context, "User with code $targetCode not found!", Toast.LENGTH_SHORT).show()
                                                    }
                                                }
                                                .addOnFailureListener { e ->
                                                    isSaving = false
                                                    Toast.makeText(context, "Error verifying friend: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                }
                                        }
                                    },
                                    shape = RoundedCornerShape(12.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = SecondaryPurple),
                                    enabled = !isSaving
                                ) {
                                    Text(stringResource(R.string.btn_add), color = Color.White)
                                }
                            }

                            if (friendsList.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Text(
                                    text = "Current Friends:",
                                    fontSize = 13.sp,
                                    color = TextSecondary,
                                    modifier = Modifier.padding(bottom = 6.dp)
                                )
                                // List Codes & Names
                                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    friendsList.forEach { fCode ->
                                        val displayName = friendsNamesMap[fCode] ?: "Loading athlete name..."
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .background(
                                                    color = Color.Black.copy(alpha = 0.2f),
                                                    shape = RoundedCornerShape(8.dp)
                                                )
                                                .border(1.dp, BorderMuted, RoundedCornerShape(8.dp))
                                                .padding(horizontal = 12.dp, vertical = 8.dp),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                text = "$displayName ($fCode)",
                                                color = Color.White,
                                                fontSize = 12.sp,
                                                maxLines = 1,
                                                modifier = Modifier.weight(1f)
                                            )
                                            Text(
                                                text = "Remove",
                                                color = Color.Red,
                                                fontSize = 11.sp,
                                                fontWeight = FontWeight.Bold,
                                                modifier = Modifier
                                                    .clickable {
                                                        isSaving = true
                                                        db.collection("users").whereEqualTo("numericId", fCode).get()
                                                            .addOnSuccessListener { querySnapshot ->
                                                                if (!querySnapshot.isEmpty) {
                                                                    val friendDoc = querySnapshot.documents.first()
                                                                    val friendUid = friendDoc.id
                                                                    
                                                                    val myRef = db.collection("users").document(uid)
                                                                    val friendRef = db.collection("users").document(friendUid)
                                                                    
                                                                    db.runTransaction { transaction ->
                                                                        val myDoc = transaction.get(myRef)
                                                                        val fDoc = transaction.get(friendRef)
                                                                        
                                                                        val myFriends = (myDoc.get("friendsList") as? List<*>)?.mapNotNull { it as? String }?.toMutableList() ?: mutableListOf()
                                                                        val friendFriends = (fDoc.get("friendsList") as? List<*>)?.mapNotNull { it as? String }?.toMutableList() ?: mutableListOf()
                                                                        
                                                                        myFriends.remove(fCode)
                                                                        friendFriends.remove(numericId)
                                                                        
                                                                        transaction.update(myRef, "friendsList", myFriends)
                                                                        transaction.update(friendRef, "friendsList", friendFriends)
                                                                    }.addOnSuccessListener {
                                                                        isSaving = false
                                                                        friendsList = friendsList.filter { it != fCode }
                                                                        Toast.makeText(context, "Friend removed!", Toast.LENGTH_SHORT).show()
                                                                    }.addOnFailureListener { e ->
                                                                        isSaving = false
                                                                        Toast.makeText(context, "Failed to remove friend: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                                    }
                                                                } else {
                                                                    val updatedList = friendsList.filter { it != fCode }
                                                                    db.collection("users").document(uid)
                                                                        .update("friendsList", updatedList)
                                                                        .addOnSuccessListener {
                                                                            isSaving = false
                                                                            friendsList = updatedList
                                                                            Toast.makeText(context, "Friend removed from list.", Toast.LENGTH_SHORT).show()
                                                                        }
                                                                        .addOnFailureListener { e ->
                                                                            isSaving = false
                                                                            Toast.makeText(context, "Failed to remove: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                                        }
                                                                }
                                                            }
                                                            .addOnFailureListener { e ->
                                                                isSaving = false
                                                                Toast.makeText(context, "Error locating friend: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                                                            }
                                                    }
                                                    .padding(4.dp)
                                            )
                                        }
                                    }
                                }
                            } else {
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = "No friends added yet. Add a friend's Code (e.g. #12345) to compete in challenges!",
                                    fontSize = 12.sp,
                                    color = TextSecondary,
                                    textAlign = TextAlign.Center,
                                    modifier = Modifier.fillMaxWidth()
                                )
                            }
                        }
                    }
                }

                // Error logs in form
                if (errorMessage != null) {
                    item {
                        Text(
                            text = errorMessage ?: "",
                            color = Color.Red,
                            fontSize = 13.sp,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                // Button Save
                item {
                    Button(
                        onClick = {
                            val weight = weightStr.toFloatOrNull()
                            val height = heightStr.toFloatOrNull()

                            if (name.isBlank() || dobTimestamp == 0L || weight == null || height == null) {
                                errorMessage = "Please enter valid profile details."
                                return@Button
                            }

                            isSaving = true
                            errorMessage = null

                            val profile = UserProfile(
                                uid = uid,
                                name = name,
                                numericId = numericId,
                                dob = com.google.firebase.Timestamp(java.util.Date(dobTimestamp)),
                                weightKg = weight,
                                heightCm = height,
                                gender = selectedGender,
                                preferredLanguage = selectedLanguage,
                                xpPoints = xpPoints,
                                level = level,
                                friendsList = friendsList,
                                createdAt = com.google.firebase.Timestamp(java.util.Date(if (createdAt == 0L) System.currentTimeMillis() else createdAt)),
                                modifiedAt = com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis())),
                                lastActive = com.google.firebase.Timestamp(java.util.Date(System.currentTimeMillis()))
                            )

                            // Update in Firestore (sets or updates)
                            db.collection("users").document(uid).set(profile)
                                .addOnSuccessListener {
                                    isSaving = false
                                    // Apply selected language immediately so Dashboard reloads in correct locale
                                    if (selectedLanguage != "SYSTEM") {
                                        AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(selectedLanguage))
                                    } else {
                                        AppCompatDelegate.setApplicationLocales(LocaleListCompat.getEmptyLocaleList())
                                    }
                                    Toast.makeText(context, "Profile saved!", Toast.LENGTH_SHORT).show()
                                    onBack()
                                }
                                .addOnFailureListener { e ->
                                    isSaving = false
                                    errorMessage = "Save failed: ${e.localizedMessage}"
                                }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
                        contentPadding = PaddingValues(),
                        enabled = !isSaving
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
                            if (isSaving) {
                                CircularProgressIndicator(color = Color(0xFF0C0F14), modifier = Modifier.size(24.dp))
                            } else {
                                Text(
                                    text = stringResource(R.string.btn_save),
                                    fontSize = 16.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFF0C0F14),
                                    letterSpacing = 1.5.sp
                                )
                            }
                        }
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }
    }
}
}
