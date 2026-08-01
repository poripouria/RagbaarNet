package com.ragbaarnet.telemetry

import android.Manifest
import android.content.Context
import android.content.res.Configuration
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import androidx.core.app.ActivityCompat
import android.view.Surface
import java.io.BufferedReader
import java.io.File
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
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
    private var previewSurface: Surface? = null
    @Volatile
    private var running = false

    companion object {
        private const val DEFAULT_PORT = 8080
        private const val TAG = "CameraWebStream"
        private const val CHUNK_CAPACITY = 256
        private const val VIDEO_BIT_RATE = 3_500_000
        private const val VIDEO_FRAME_RATE = 30
    }

    fun start(previewSurface: Surface? = null) {
        if (running) {
            return
        }

        this.previewSurface = previewSurface
        running = true
        
        // Start camera session immediately
        synchronized(sessionLock) {
            activeSession = CameraStreamSession(context, cameraManager, previewSurface).also { it.start() }
        }

        serverSocket = ServerSocket(port)
        acceptThread = Thread { acceptLoop() }.apply {
            name = "CameraWebStream-Accept"
            isDaemon = true
            start()
        }

        onStatus("HTTP stream listening on port $port")
    }

    fun stop() {
        running = false

        try {
            serverSocket?.close()
        } catch (_: IOException) {
        }

        serverSocket = null
        acceptThread = null

        synchronized(sessionLock) {
            activeSession?.stop()
            activeSession = null
        }

        onStatus("HTTP stream stopped")
    }

    fun getStreamPath(): String = "/stream.webm"

    fun getStreamUrl(host: String): String = "http://$host:$port${getStreamPath()}"

    private var activeSession: CameraStreamSession? = null

    private fun acceptLoop() {
        while (running) {
            val socket = try {
                serverSocket?.accept()
            } catch (acceptErr: IOException) {
                if (running) {
                    Log.e(TAG, "Accept failed", acceptErr)
                }
                null
            } ?: break

            try {
                handleClient(socket)
            } catch (clientErr: Exception) {
                Log.e(TAG, "Client handler failed", clientErr)
                try {
                    socket.close()
                } catch (_: IOException) {
                }
            }
        }
    }

    private fun handleClient(socket: Socket) {
        socket.soTimeout = 15_000
        val reader = BufferedReader(InputStreamReader(socket.getInputStream(), StandardCharsets.US_ASCII))
        val requestLine = reader.readLine() ?: return
        val requestParts = requestLine.split(" ")
        if (requestParts.size < 2) {
            respondText(socket, 400, "Bad Request", "Malformed request")
            return
        }

        var line = reader.readLine()
        while (!line.isNullOrEmpty()) {
            line = reader.readLine()
        }

        val path = requestParts[1]
        when (path) {
            "/", "/health" -> respondText(socket, 200, "OK", "RagbaarTelemetry stream server is running")
            getStreamPath() -> streamVideo(socket)
            else -> respondText(socket, 404, "Not Found", "Unknown path")
        }
    }

    private fun streamVideo(socket: Socket) {
        val session = synchronized(sessionLock) { activeSession }
        if (session == null) {
            respondText(socket, 503, "Service Unavailable", "Camera session not initialized")
            return
        }
        
        val output = socket.getOutputStream()
        val headers = buildString {
            append("HTTP/1.1 200 OK\r\n")
            append("Content-Type: ${session.getMimeType()}\r\n")
            append("Cache-Control: no-store, no-cache, must-revalidate, proxy-revalidate\r\n")
            append("Pragma: no-cache\r\n")
            append("Connection: close\r\n")
            append("Transfer-Encoding: chunked\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("\r\n")
        }

        output.write(headers.toByteArray(StandardCharsets.US_ASCII))
        output.flush()

        try {
            session.streamTo(output)
        } finally {
            try {
                writeChunkTerminator(output)
            } catch (_: IOException) {
            }
            try {
                socket.close()
            } catch (_: IOException) {
            }
        }
    }

    private fun respondText(socket: Socket, code: Int, status: String, body: String) {
        val payload = body.toByteArray(StandardCharsets.UTF_8)
        val response = buildString {
            append("HTTP/1.1 $code $status\r\n")
            append("Content-Type: text/plain; charset=utf-8\r\n")
            append("Content-Length: ${payload.size}\r\n")
            append("Connection: close\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("\r\n")
        }

        socket.getOutputStream().use { output ->
            output.write(response.toByteArray(StandardCharsets.US_ASCII))
            output.write(payload)
            output.flush()
        }
        socket.close()
    }

    private fun writeChunkTerminator(output: OutputStream) {
        output.write("0\r\n\r\n".toByteArray(StandardCharsets.US_ASCII))
        output.flush()
    }

    private class CameraStreamSession(
        private val context: Context,
        private val cameraManager: CameraManager,
        private val previewSurface: Surface? = null
    ) {
        private val chunkQueue = LinkedBlockingQueue<ByteArray>(CHUNK_CAPACITY)
        private val readyLatch = CountDownLatch(1)
        private val cameraThread = HandlerThread("CameraWebStream-Camera")
        private var cameraHandler: Handler? = null
        private var cameraDevice: CameraDevice? = null
        private var captureSession: CameraCaptureSession? = null
        private var mediaRecorder: MediaRecorder? = null
        private var pumpThread: Thread? = null
        private var outputFile: File? = null

        @Volatile
        private var running = false

        @Volatile
        private var startError: Throwable? = null

        @Volatile
        private var mimeType = "video/webm"

        fun start() {
            if (running) {
                return
            }

            if (ActivityCompat.checkSelfPermission(context, Manifest.permission.CAMERA) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                throw SecurityException("Camera permission is required for streaming")
            }

            running = true
            cameraThread.start()
            cameraHandler = Handler(cameraThread.looper)

            try {
                prepareOutputFile()
                openCamera()
            } catch (err: Exception) {
                startError = err
                readyLatch.countDown()
            }

            if (!readyLatch.await(15, TimeUnit.SECONDS)) {
                stop()
                throw IOException("Timed out waiting for camera stream startup")
            }

            startError?.let {
                stop()
                throw it as? Exception ?: IOException(it)
            }
        }

        fun streamTo(output: OutputStream) {
            while (running) {
                val chunk = try {
                    chunkQueue.poll(1, TimeUnit.SECONDS)
                } catch (_: InterruptedException) {
                    null
                }

                if (chunk == null) {
                    if (!running) {
                        break
                    }
                    continue
                }

                writeChunk(output, chunk)
            }
        }

        fun stop() {
            if (!running) {
                releaseResources()
                return
            }

            running = false

            try {
                captureSession?.close()
            } catch (_: Exception) {
            }
            captureSession = null

            try {
                cameraDevice?.close()
            } catch (_: Exception) {
            }
            cameraDevice = null

            try {
                mediaRecorder?.stop()
            } catch (_: Exception) {
            }

            releaseResources()

            if (cameraThread.isAlive) {
                cameraThread.quitSafely()
            }
        }

        fun getMimeType(): String = mimeType

        private fun prepareOutputFile() {
            val cacheDir = context.cacheDir ?: context.filesDir
            outputFile = File.createTempFile("ragbaarnet_telemetry_stream_", ".webm", cacheDir).apply {
                if (exists()) {
                    delete()
                }
                createNewFile()
            }

            mediaRecorder = buildRecorder(outputFile!!)
            startPumpThread(outputFile!!)
        }

        private fun buildRecorder(outputFile: File): MediaRecorder {
            val recorder = MediaRecorder()
            val portrait = context.resources.configuration.orientation == Configuration.ORIENTATION_PORTRAIT
            val outputCandidates = listOf(
                RecorderConfig("video/webm", MediaRecorder.OutputFormat.WEBM, MediaRecorder.VideoEncoder.VP8),
                RecorderConfig("video/mp4", MediaRecorder.OutputFormat.MPEG_4, MediaRecorder.VideoEncoder.H264)
            )

            var lastError: Throwable? = null
            for (candidate in outputCandidates) {
                try {
                    recorder.reset()
                    recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
                    recorder.setOutputFormat(candidate.outputFormat)
                    recorder.setVideoEncoder(candidate.videoEncoder)
                    recorder.setVideoEncodingBitRate(VIDEO_BIT_RATE)
                    recorder.setVideoFrameRate(VIDEO_FRAME_RATE)

                    val targetSize = chooseVideoSize()
                    recorder.setVideoSize(targetSize.width, targetSize.height)
                    recorder.setOutputFile(outputFile)
                    recorder.setOrientationHint(if (portrait) 90 else 0)
                    recorder.prepare()
                    mimeType = candidate.mimeType
                    return recorder
                } catch (err: Throwable) {
                    lastError = err
                }
            }

            try {
                recorder.release()
            } catch (_: Exception) {
            }

            throw IOException("Unable to configure camera recorder", lastError)
        }

        private fun chooseVideoSize(): Size {
            val cameraId = findBackCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val sizes = map?.getOutputSizes(MediaRecorder::class.java)?.toList().orEmpty()

            val preferred = sizes.firstOrNull { it.width == 1280 && it.height == 720 }
                ?: sizes.firstOrNull { it.width == 1920 && it.height == 1080 }
                ?: sizes.firstOrNull { it.width > it.height }

            return preferred ?: Size(1280, 720)
        }

        private fun findBackCameraId(): String {
            for (cameraId in cameraManager.cameraIdList) {
                val characteristics = cameraManager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                    return cameraId
                }
            }

            return cameraManager.cameraIdList.first()
        }

        private fun openCamera() {
            val cameraId = findBackCameraId()
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    cameraDevice = device
                    try {
                        startCaptureSession(device)
                    } catch (err: Throwable) {
                        startError = err
                        readyLatch.countDown()
                    }
                }

                override fun onDisconnected(device: CameraDevice) {
                    startError = IOException("Camera disconnected")
                    readyLatch.countDown()
                    device.close()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    startError = IOException("Camera error $error")
                    readyLatch.countDown()
                    device.close()
                }
            }, cameraHandler)
        }

        private fun startCaptureSession(device: CameraDevice) {
            val recorderSurface = mediaRecorder?.surface ?: throw IOException("Recorder surface unavailable")
            val requestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
            requestBuilder.addTarget(recorderSurface)
            
            val surfaces = mutableListOf(recorderSurface)
            previewSurface?.let {
                requestBuilder.addTarget(it)
                surfaces.add(it)
            }
            
            requestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)

            device.createCaptureSession(
                surfaces,
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        try {
                            mediaRecorder?.start()
                            session.setRepeatingRequest(requestBuilder.build(), null, cameraHandler)
                            readyLatch.countDown()
                        } catch (err: Throwable) {
                            startError = err
                            readyLatch.countDown()
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        startError = IOException("Camera capture session configuration failed")
                        readyLatch.countDown()
                    }
                },
                cameraHandler
            )
        }

        private fun startPumpThread(recordingFile: File) {
            pumpThread = Thread {
                var position = 0L
                while (running) {
                    if (!recordingFile.exists()) {
                        Thread.sleep(50)
                        continue
                    }

                    val length = recordingFile.length()
                    if (length <= position) {
                        Thread.sleep(50)
                        continue
                    }

                    RandomAccessFile(recordingFile, "r").use { raf ->
                        raf.seek(position)
                        val buffer = ByteArray(32 * 1024)
                        while (running) {
                            val read = raf.read(buffer)
                            if (read <= 0) {
                                break
                            }

                            position += read.toLong()
                            chunkQueue.put(buffer.copyOf(read))

                            if (raf.filePointer >= recordingFile.length()) {
                                break
                            }
                        }
                    }
                }
            }.apply {
                name = "CameraWebStream-Pump"
                isDaemon = true
                start()
            }
        }

        private fun writeChunk(output: OutputStream, chunk: ByteArray) {
            val sizeLine = chunk.size.toString(16).toByteArray(StandardCharsets.US_ASCII)
            output.write(sizeLine)
            output.write("\r\n".toByteArray(StandardCharsets.US_ASCII))
            output.write(chunk)
            output.write("\r\n".toByteArray(StandardCharsets.US_ASCII))
            output.flush()
        }

        private fun releaseResources() {
            try {
                outputFile?.delete()
            } catch (_: Exception) {
            }
            outputFile = null

            try {
                mediaRecorder?.release()
            } catch (_: Exception) {
            }
            mediaRecorder = null

            try {
                pumpThread?.interrupt()
            } catch (_: Exception) {
            }
            pumpThread = null
        }

        private data class RecorderConfig(
            val mimeType: String,
            val outputFormat: Int,
            val videoEncoder: Int
        )
    }
}