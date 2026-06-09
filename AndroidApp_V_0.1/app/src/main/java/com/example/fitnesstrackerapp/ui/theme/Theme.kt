package com.example.fitnesstrackerapp.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val DarkBackground = Color(0xFF0C0F14)
val DarkSurface = Color(0xFF17202A)
val PrimaryCyan = Color(0xFF00F0FF)
val SecondaryPurple = Color(0xFFA020F0)
val AccentGreen = Color(0xFF39FF14)
val TextPrimary = Color(0xFFFFFFFF)
val TextSecondary = Color(0xFF9EADB8)
val BorderMuted = Color(0xFF2C3A47)

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryCyan,
    secondary = SecondaryPurple,
    background = DarkBackground,
    surface = DarkSurface,
    onPrimary = Color(0xFF0C0F14),
    onSecondary = Color.White,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    outline = BorderMuted
)

@Composable
fun FitnessTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}
