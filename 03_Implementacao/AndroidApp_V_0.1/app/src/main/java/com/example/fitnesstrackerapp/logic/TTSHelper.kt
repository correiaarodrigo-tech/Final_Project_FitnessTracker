package com.example.fitnesstrackerapp.logic

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * TextToSpeech engine helper for fitness coaching cues in Portuguese.
 * Implements a 0.5-second debounce for dynamic feedback, while allowing
 * rep counts and timer milestones to speak immediately.
 */
class TTSHelper(context: Context) : TextToSpeech.OnInitListener {
    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isInitialized = false
    private var lastSpokenCue = ""
    private var lastSpeakTime = 0L

    private val mainHandler = Handler(Looper.getMainLooper())
    private var pendingRunnable: Runnable? = null

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val ptLocale = Locale("pt", "PT")
            val result = tts?.setLanguage(ptLocale)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.w("TTSHelper", "Portuguese language not supported; falling back to system default")
                tts?.setLanguage(Locale.getDefault())
            }
            isInitialized = true
            Log.d("TTSHelper", "TextToSpeech successfully initialized in Portuguese")
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

        // Cancel previous pending speech
        pendingRunnable?.let { mainHandler.removeCallbacks(it) }

        if (overrideCooldown) {
            // Rep counts or countdown progress speak immediately
            speakImmediate(text, currentTime)
        } else {
            // Debounce standard posture instructions by 500ms
            val runnable = Runnable {
                speakImmediate(text, System.currentTimeMillis())
            }
            pendingRunnable = runnable
            mainHandler.postDelayed(runnable, 500)
        }
    }

    private fun speakImmediate(text: String, currentTime: Long) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        lastSpokenCue = text
        lastSpeakTime = currentTime
    }

    fun shutdown() {
        mainHandler.removeCallbacksAndMessages(null)
        tts?.stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}
