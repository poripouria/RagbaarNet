package com.ragbaarnet.telemetry

import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.Executors

class MjpegServer(private val port: Int) {
    private var serverSocket: ServerSocket? = null
    private var isRunning = false
    private val executor = Executors.newCachedThreadPool()
    private var lastJpeg: ByteArray? = null
    private val clients = mutableListOf<Socket>()

    fun start() {
        if (isRunning) return
        isRunning = true
        executor.execute {
            try {
                serverSocket = ServerSocket(port)
                Log.d("MjpegServer", "Server started on port $port")
                while (isRunning) {
                    val socket = serverSocket?.accept()
                    socket?.let {
                        Log.d("MjpegServer", "New client connected: ${it.inetAddress}")
                        executor.execute { handleClient(it) }
                    }
                }
            } catch (e: Exception) {
                Log.e("MjpegServer", "Server error: ${e.message}")
            }
        }
    }

    fun stop() {
        isRunning = false
        serverSocket?.close()
        executor.shutdownNow()
    }

    fun updateFrame(jpeg: ByteArray) {
        lastJpeg = jpeg
    }

    private fun handleClient(socket: Socket) {
        try {
            val inputStream = socket.getInputStream()
            val reader = inputStream.bufferedReader()
            val requestLine = reader.readLine() ?: return
            
            val outputStream = socket.getOutputStream()
            
            // Handle CORS Pre-flight
            if (requestLine.startsWith("OPTIONS")) {
                outputStream.write(("HTTP/1.1 204 No Content\r\n" +
                        "Access-Control-Allow-Origin: *\r\n" +
                        "Access-Control-Allow-Methods: GET, OPTIONS\r\n" +
                        "Access-Control-Allow-Headers: *\r\n" +
                        "\r\n").toByteArray())
                outputStream.flush()
                return
            }

            val boundary = "frame"
            
            // HTTP Header for MJPEG with CORS support
            outputStream.write(("HTTP/1.0 200 OK\r\n" +
                    "Server: RagbaarTelemetry\r\n" +
                    "Connection: close\r\n" +
                    "Max-Age: 0\r\n" +
                    "Expires: 0\r\n" +
                    "Cache-Control: no-cache, private\r\n" +
                    "Pragma: no-cache\r\n" +
                    "Access-Control-Allow-Origin: *\r\n" +
                    "Content-Type: multipart/x-mixed-replace; boundary=$boundary\r\n" +
                    "\r\n").toByteArray())

            while (isRunning && !socket.isClosed) {
                val frame = lastJpeg
                if (frame != null) {
                    outputStream.write(("--$boundary\r\n" +
                            "Content-Type: image/jpeg\r\n" +
                            "Content-Length: ${frame.size}\r\n" +
                            "\r\n").toByteArray())
                    outputStream.write(frame)
                    outputStream.write("\r\n".toByteArray())
                    outputStream.flush()
                }
                Thread.sleep(30) // ~30 FPS
            }
        } catch (e: Exception) {
            Log.d("MjpegServer", "Client disconnected: ${e.message}")
        } finally {
            socket.close()
        }
    }
}
