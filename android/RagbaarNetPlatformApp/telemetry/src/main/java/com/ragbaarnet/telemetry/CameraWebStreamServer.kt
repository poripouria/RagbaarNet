package com.ragbaarnet.telemetry

import android.Manifest
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import androidx.core.app.ActivityCompat
import android.view.Surface
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class CameraWebStreamServer(
    private val context: Context,
    private val port: Int = DEFAULT_PORT,
    private val onStatus: (String) -> Unit = {}
) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val sessionLock = Any()
    private var serverSocket: ServerSocket? = null
    private var acceptThread: Thread? = null
    
    @Volatile
    private var running = false
    
    private val clients = Collections.synchronizedSet(mutableSetOf<Socket>())

    companion object {
        private const val DEFAULT_PORT = 8080
        private const val TAG = "CameraWebStream"
        private const val STREAM_WIDTH = 1280
        private const val STREAM_HEIGHT = 720
    }

    fun start(previewSurface: Surface? = null) {
        if (running) return
        running = true
        
        synchronized(sessionLock) {
            activeSession = CameraStreamSession(context, cameraManager, previewSurface) { jpeg ->
                broadcastFrame(jpeg)
            }.also { it.start() }
        }

        serverSocket = ServerSocket(port)
        acceptThread = Thread { acceptLoop() }.apply {
            name = "CameraWebStream-Accept"
            isDaemon = true
            start()
        }

        onStatus("MJPEG stream listening on port $port")
    }

    fun stop() {
        running = false
        try { serverSocket?.close() } catch (_: IOException) {}
        serverSocket = null
        acceptThread = null

        synchronized(clients) {
            clients.forEach { try { it.close() } catch (_: Exception) {} }
            clients.clear()
        }

        synchronized(sessionLock) {
            activeSession?.stop()
            activeSession = null
        }
        onStatus("MJPEG stream stopped")
    }

    fun getStreamPath(): String = "/stream.webm" // Kept for compatibility with your UI
    fun getStreamUrl(host: String): String = "http://$host:$port${getStreamPath()}"

    private var activeSession: CameraStreamSession? = null

    private fun acceptLoop() {
        while (running) {
            val socket = try {
                serverSocket?.accept()
            } catch (e: IOException) {
                if (running) Log.e(TAG, "Accept failed", e)
                null
            } ?: break

            Thread { handleClient(socket) }.start()
        }
    }

    private fun handleClient(socket: Socket) {
        try {
            socket.soTimeout = 10000
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), StandardCharsets.US_ASCII))
            val requestLine = reader.readLine() ?: return
            val requestParts = requestLine.split(" ")
            if (requestParts.size < 2) {
                respondText(socket, 400, "Bad Request", "Malformed")
                return
            }

            val path = requestParts[1]
            if (path == getStreamPath()) {
                val output = socket.getOutputStream()
                val boundary = "frame"
                val headers = buildString {
                    append("HTTP/1.1 200 OK\r\n")
                    append("Content-Type: multipart/x-mixed-replace; boundary=$boundary\r\n")
                    append("Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n")
                    append("Pragma: no-cache\r\n")
                    append("Connection: close\r\n")
                    append("Access-Control-Allow-Origin: *\r\n")
                    append("\r\n")
                }
                output.write(headers.toByteArray(StandardCharsets.US_ASCII))
                output.flush()
                
                synchronized(clients) { clients.add(socket) }
                
                // Keep the thread alive while the socket is open
                while (running && !socket.isClosed) {
                    Thread.sleep(1000)
                }
            } else {
                respondText(socket, 200, "OK", "RagbaarTelemetry MJPEG Server")
            }
        } catch (e: Exception) {
            Log.d(TAG, "Client disconnected: ${e.message}")
        } finally {
            synchronized(clients) { clients.remove(socket) }
            try { socket.close() } catch (_: Exception) {}
        }
    }

    private fun broadcastFrame(jpeg: ByteArray) {
        val boundary = "frame"
        val header = buildString {
            append("--$boundary\r\n")
            append("Content-Type: image/jpeg\r\n")
            append("Content-Length: ${jpeg.size}\r\n")
            append("\r\n")
        }.toByteArray(StandardCharsets.US_ASCII)
        val footer = "\r\n".toByteArray(StandardCharsets.US_ASCII)

        synchronized(clients) {
            val iterator = clients.iterator()
            while (iterator.hasNext()) {
                val socket = iterator.next()
                try {
                    val output = socket.getOutputStream()
                    output.write(header)
                    output.write(jpeg)
                    output.write(footer)
                    output.flush()
                } catch (e: Exception) {
                    iterator.remove()
                    try { socket.close() } catch (_: Exception) {}
                }
            }
        }
    }

    private fun respondText(socket: Socket, code: Int, status: String, body: String) {
        try {
            val payload = body.toByteArray(StandardCharsets.UTF_8)
            val response = "HTTP/1.1 $code $status\r\nContent-Length: ${payload.size}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
            socket.getOutputStream().apply {
                write(response.toByteArray(StandardCharsets.US_ASCII))
                write(payload)
                flush()
            }
        } catch (_: Exception) {} finally { try { socket.close() } catch (_: Exception) {} }
    }

    private class CameraStreamSession(
        private val context: Context,
        private val cameraManager: CameraManager,
        private val previewSurface: Surface? = null,
        private val onFrame: (ByteArray) -> Unit
    ) {
        private val readyLatch = CountDownLatch(1)
        private val cameraThread = HandlerThread("CameraMJPEG-Camera")
        private var cameraHandler: Handler? = null
        private var cameraDevice: CameraDevice? = null
        private var captureSession: CameraCaptureSession? = null
        private var imageReader: ImageReader? = null
        
        @Volatile
        private var running = false
        
        fun start() {
            if (running) return
            running = true
            cameraThread.start()
            cameraHandler = Handler(cameraThread.looper)
            openCamera()
            readyLatch.await(5, TimeUnit.SECONDS)
        }

        fun stop() {
            running = false
            try { captureSession?.close() } catch (_: Exception) {}
            try { cameraDevice?.close() } catch (_: Exception) {}
            try { imageReader?.close() } catch (_: Exception) {}
            cameraThread.quitSafely()
        }

        private fun openCamera() {
            val cameraId = findBackCameraId()
            if (ActivityCompat.checkSelfPermission(context, Manifest.permission.CAMERA) != android.content.pm.PackageManager.PERMISSION_GRANTED) return
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    cameraDevice = device
                    startCaptureSession(device)
                }
                override fun onDisconnected(device: CameraDevice) = device.close()
                override fun onError(device: CameraDevice, error: Int) = device.close()
            }, cameraHandler)
        }

        private fun startCaptureSession(device: CameraDevice) {
            imageReader = ImageReader.newInstance(STREAM_WIDTH, STREAM_HEIGHT, ImageFormat.YUV_420_888, 2)
            imageReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val jpeg = yuvToJpeg(image)
                    onFrame(jpeg)
                } catch (e: Exception) {
                    Log.e(TAG, "JPEG compression failed", e)
                } finally {
                    image.close()
                }
            }, cameraHandler)

            val surfaces = mutableListOf(imageReader!!.surface)
            previewSurface?.let { surfaces.add(it) }

            device.createCaptureSession(surfaces, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(imageReader!!.surface)
                        previewSurface?.let { addTarget(it) }
                    }
                    session.setRepeatingRequest(request.build(), null, cameraHandler)
                    readyLatch.countDown()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) = readyLatch.countDown()
            }, cameraHandler)
        }

        private fun yuvToJpeg(image: Image): ByteArray {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val out = ByteArrayOutputStream()
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 70, out)
            return out.toByteArray()
        }

        private fun findBackCameraId(): String {
            for (id in cameraManager.cameraIdList) {
                if (cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK) return id
            }
            return cameraManager.cameraIdList.first()
        }
    }
}
