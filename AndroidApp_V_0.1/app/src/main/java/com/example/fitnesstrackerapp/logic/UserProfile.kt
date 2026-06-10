package com.example.fitnesstrackerapp.logic

import com.google.firebase.Timestamp
import java.util.Date

data class UserProfile(
    val uid: String = "",
    val name: String = "",
    val numericId: String = "",       // 5-digit ID like #12345
    val dob: Any? = null,             // Can be Long (legacy) or Timestamp (new)
    val weightKg: Float = 0f,
    val heightCm: Float = 0f,
    val gender: String = "OTHER",     // "MALE", "FEMALE", "OTHER"
    val xpPoints: Int = 0,
    val level: Int = 1,
    val friendsList: List<String> = emptyList(), // Store friend numericIds (e.g. "#12345")
    val createdAt: Any? = null,       // Can be Long (legacy) or Timestamp (new)
    val modifiedAt: Any? = null       // Can be Long (legacy) or Timestamp (new)
) {
    // Safely resolve fields to Long milliseconds
    fun getDobLong(): Long = toMillis(dob)
    fun getCreatedAtLong(): Long = toMillis(createdAt)
    fun getModifiedAtLong(): Long = toMillis(modifiedAt)

    private fun toMillis(value: Any?): Long {
        return when (value) {
            is Long -> value
            is Double -> value.toLong()
            is Timestamp -> value.toDate().time
            is Date -> value.time
            is Map<*, *> -> {
                val seconds = value["seconds"] as? Long ?: 0L
                val nanoseconds = value["nanoseconds"] as? Long ?: 0L
                seconds * 1000 + (nanoseconds / 1000000)
            }
            else -> 0L
        }
    }
}
