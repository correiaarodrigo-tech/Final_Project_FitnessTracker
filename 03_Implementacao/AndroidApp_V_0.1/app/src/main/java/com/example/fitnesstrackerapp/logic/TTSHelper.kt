package com.example.fitnesstrackerapp.logic

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * TextToSpeech engine helper for fitness coaching cues.
 * Rate-limits speech to avoid overlapping, with an override for rep completion.
 */
class TTSHelper(context: Context) : TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isInitialized = false
    private var lastSpokenCue = ""
    private var lastSpeakTime = 0L

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTSHelper", "US English is not supported or missing speech data")
            } else {
                isInitialized = true
                Log.d("TTSHelper", "TextToSpeech successfully initialized")
            }
        } else {
            Log.e("TTSHelper", "TextToSpeech initialization failed")
        }
    }

    fun speak(text: String, overrideCooldown: Boolean = false) {
        if (!isInitialized) return
        val currentTime = System.currentTimeMillis()
        
        // Cooldown of 4 seconds to avoid verbal clutter, unless it is a rep trigger
        if (!overrideCooldown && text == lastSpokenCue && (currentTime - lastSpeakTime) < 4000) {
            return
        }

        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        lastSpokenCue = text
        lastSpeakTime = currentTime
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
