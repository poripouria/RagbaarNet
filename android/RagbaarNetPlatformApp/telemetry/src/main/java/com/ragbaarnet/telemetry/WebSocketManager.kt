package com.ragbaarnet.telemetry

import android.util.Log
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit

class WebSocketManager(private val serverUrl: String) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var isConnected = false
    private var shouldReconnect = true

    interface ConnectionListener {
        fun onConnected()
        fun onDisconnected()
        fun onError(error: String)
    }

    var listener: ConnectionListener? = null

    fun connect() {
        shouldReconnect = true
        val request = Request.Builder().url(serverUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "Connected to $serverUrl")
                isConnected = true
                listener?.onConnected()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Handle incoming control messages if needed
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                isConnected = false
                Log.d("WebSocket", "Closing: $reason")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                isConnected = false
                listener?.onDisconnected()
                if (shouldReconnect) reconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                Log.e("WebSocket", "Failure: ${t.message}")
                listener?.onError(t.message ?: "Unknown error")
                if (shouldReconnect) reconnect()
            }
        })
    }

    fun disconnect() {
        shouldReconnect = false
        webSocket?.close(1000, "App closed")
        isConnected = false
    }

    private fun reconnect() {
        Log.d("WebSocket", "Attempting reconnect...")
        Thread.sleep(2000)
        if (shouldReconnect) connect()
    }

    fun sendFrame(jpegBytes: ByteArray) {
        if (isConnected) {
            webSocket?.send(jpegBytes.toByteString())
        }
    }

    fun sendTelemetry(json: String) {
        if (isConnected) {
            webSocket?.send(json)
        }
    }

    fun isConnected(): Boolean = isConnected
}
