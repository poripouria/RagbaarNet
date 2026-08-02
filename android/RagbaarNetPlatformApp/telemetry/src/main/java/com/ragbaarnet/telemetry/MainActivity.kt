package com.ragbaarnet.telemetry

import android.Manifest
import android.animation.ObjectAnimator
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.view.ScaleGestureDetector
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.ImageButton
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import android.view.ViewGroup.MarginLayoutParams
import com.google.android.material.button.MaterialButton
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var speedSeekBar: SeekBar
    private lateinit var rpmSeekBar: SeekBar
    private lateinit var accelSeekBar: SeekBar
    private lateinit var speedValueText: TextView
    private lateinit var rpmValueText: TextView
    private lateinit var accelValueText: TextView
    private lateinit var statusText: TextView
    private lateinit var cameraOverlay: View
    private lateinit var viewFinder: TextureView
    private lateinit var streamToggleButton: MaterialButton
    
    private lateinit var scaleGestureDetector: ScaleGestureDetector

    private lateinit var pillSelector: View
    private lateinit var textAuto: TextView
    private lateinit var textManual: TextView

    private var isAutoMode = true
    private var isStreaming = false
    private val handler = Handler(Looper.getMainLooper())
    private val telemetryHttpClient = OkHttpClient()
    private var cameraWebStreamServer: CameraWebStreamServer? = null
    @Volatile
    private var telemetryOnline = false
    @Volatile
    private var telemetryRequestInFlight = false

    // Sensors
    private lateinit var sensorManager: SensorManager
    private var linearAccelSensor: Sensor? = null
    private var currentLinearAccel: Float? = null

    // Location
    private lateinit var locationManager: LocationManager
    private var currentGpsSpeed: Float? = null

    private val sendTelemetryRunnable = object : Runnable {
        override fun run() {
            broadcastTelemetry()
            handler.postDelayed(this, 250)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        speedSeekBar = findViewById(R.id.speedSeekBar)
        rpmSeekBar = findViewById(R.id.rpmSeekBar)
        accelSeekBar = findViewById(R.id.accelSeekBar)
        speedValueText = findViewById(R.id.speedValueText)
        rpmValueText = findViewById(R.id.rpmValueText)
        accelValueText = findViewById(R.id.accelValueText)
        statusText = findViewById(R.id.statusText)
        cameraOverlay = findViewById(R.id.cameraOverlay)
        viewFinder = findViewById(R.id.viewFinder)
        streamToggleButton = findViewById(R.id.streamToggleButton)
        
        viewFinder.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                configureTransform(width, height)
                if (isStreaming) {
                    startStreamingWithSurface(Surface(surface))
                }
            }
            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                configureTransform(width, height)
            }
            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
        }
        
        pillSelector = findViewById(R.id.pillSelector)
        textAuto = findViewById(R.id.textAuto)
        textManual = findViewById(R.id.textManual)

        val titleText = findViewById<TextView>(R.id.titleText)
        val settingsButton = findViewById<View>(R.id.settingsButton)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(android.R.id.content)) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            
            titleText.updateLayoutParams<MarginLayoutParams> {
                topMargin = systemBars.top + (8 * resources.displayMetrics.density).toInt()
            }
            settingsButton.updateLayoutParams<MarginLayoutParams> {
                topMargin = systemBars.top + (8 * resources.displayMetrics.density).toInt()
            }
            statusText.updateLayoutParams<MarginLayoutParams> {
                bottomMargin = systemBars.bottom + (32 * resources.displayMetrics.density).toInt()
            }
            
            insets
        }

        settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        val modeClickListener = View.OnClickListener { v ->
            val newMode = (v.id == R.id.textAuto)
            if (newMode != isAutoMode) {
                isAutoMode = newMode
                animatePill(isAutoMode)
                updateUiMode()
            }
        }
        textAuto.setOnClickListener(modeClickListener)
        textManual.setOnClickListener(modeClickListener)

        streamToggleButton.setOnClickListener {
            toggleStreaming(!isStreaming)
        }

        setupZoomGestures()
        logCameraFocalLengths()
        setupSeekBarListeners()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        linearAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        updateUiMode()
        checkPermissions()
        initTelemetryTransport()
        handler.post(sendTelemetryRunnable)
    }

    private fun initTelemetryTransport() {
        telemetryOnline = false
        updateStatusDisplay()
    }

    private fun toggleStreaming(enable: Boolean) {
        if (enable) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_PERMISSIONS)
                return
            }
            
            isStreaming = true
            streamToggleButton.text = "Stream: ON"
            streamToggleButton.setTextColor(Color.BLACK)
            streamToggleButton.setBackgroundColor(Color.parseColor("#00ff88"))
            streamToggleButton.setStrokeColorResource(android.R.color.transparent)
            
            cameraOverlay.visibility = View.VISIBLE
            viewFinder.visibility = View.VISIBLE
            
            if (viewFinder.isAvailable) {
                val prefs = getSharedPreferences("TelemetryPrefs", MODE_PRIVATE)
                val initialZoom = prefs.getFloat("preferred_zoom", 0.5f)
                startStreamingWithSurface(Surface(viewFinder.surfaceTexture), initialZoom)
            }
        } else {
            isStreaming = false
            streamToggleButton.text = "Stream: OFF"
            streamToggleButton.setTextColor(Color.WHITE)
            streamToggleButton.setBackgroundColor(Color.TRANSPARENT)
            streamToggleButton.setStrokeColor(ContextCompat.getColorStateList(this, android.R.color.white))
            
            cameraOverlay.visibility = View.GONE
            viewFinder.visibility = View.GONE
            
            cameraWebStreamServer?.stop()
            cameraWebStreamServer = null

            updateStatusDisplay()
        }
    }

    private fun startStreamingWithSurface(surface: Surface, initialZoom: Float = 1.0f) {
        try {
            if (cameraWebStreamServer == null) {
                cameraWebStreamServer = CameraWebStreamServer(this) { message ->
                    runOnUiThread {
                        setStatusText(message)
                        updateStatusDisplay()
                    }
                }
                cameraWebStreamServer?.start(surface, initialZoom)
            }

            updateStatusDisplay()
            val streamUrl = cameraWebStreamServer?.getStreamUrl(getLocalIpAddress() ?: "127.0.0.1")
            Toast.makeText(this, "Stream URL: $streamUrl", Toast.LENGTH_LONG).show()
        } catch (startErr: Exception) {
            isStreaming = false
            streamToggleButton.text = "Stream: OFF"
            streamToggleButton.setTextColor(Color.WHITE)
            streamToggleButton.setBackgroundColor(Color.TRANSPARENT)
            streamToggleButton.setStrokeColor(ContextCompat.getColorStateList(this, android.R.color.white))
            cameraOverlay.visibility = View.GONE
            viewFinder.visibility = View.GONE
            cameraWebStreamServer?.stop()
            cameraWebStreamServer = null
            setStatusText("Stream failed to start: ${startErr.message ?: "unknown error"}")
            updateStatusDisplay()
            Toast.makeText(this, "Stream failed to start", Toast.LENGTH_SHORT).show()
        }
    }

    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        val rotation = windowManager.defaultDisplay.rotation
        val matrix = Matrix()
        val viewRect = android.graphics.RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        
        // Use a standard camera aspect ratio (e.g., 16:9 for the 1280x720 picked by server)
        // Note: In a real app, you'd get this from the camera characteristics
        val bufferWidth = 1280f
        val bufferHeight = 720f
        
        val bufferRect = android.graphics.RectF(0f, 0f, bufferHeight, bufferWidth)
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()

        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
            val scale = Math.max(
                viewHeight.toFloat() / bufferHeight,
                viewWidth.toFloat() / bufferWidth
            )
            matrix.postScale(scale, scale, centerX, centerY)
            matrix.postRotate((90 * (rotation - 2)).toFloat(), centerX, centerY)
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180f, centerX, centerY)
        }
        
        // Simple Center Crop for Portrait
        if (rotation == Surface.ROTATION_0 || rotation == Surface.ROTATION_180) {
            val previewAspect = bufferHeight / bufferWidth // 720/1280
            val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()
            
            var scaleX = 1f
            var scaleY = 1f
            
            if (viewAspect > previewAspect) {
                scaleY = viewAspect / previewAspect
            } else {
                scaleX = previewAspect / viewAspect
            }
            matrix.postScale(scaleX, scaleY, centerX, centerY)
        }

        viewFinder.setTransform(matrix)
    }

    private fun getLocalIpAddress(): String? {
        try {
            val en = NetworkInterface.getNetworkInterfaces()
            while (en.hasMoreElements()) {
                val intf = en.nextElement()
                val enumIpAddr = intf.inetAddresses
                while (enumIpAddr.hasMoreElements()) {
                    val inetAddress = enumIpAddr.nextElement()
                    if (!inetAddress.isLoopbackAddress && inetAddress is Inet4Address) {
                        return inetAddress.hostAddress
                    }
                }
            }
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
        return null
    }

    private fun logCameraFocalLengths() {
        val cm = getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
        try {
            for (id in cm.cameraIdList) {
                val chars = cm.getCameraCharacteristics(id)
                val facing = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
                val focalLengths = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                Log.d("CameraDebug", "Camera ID: $id, Facing: $facing, Focal Lengths: ${focalLengths?.joinToString()}")
            }
        } catch (e: Exception) {
            Log.e("CameraDebug", "Error logging cameras", e)
        }
    }

    private fun setupZoomGestures() {
        scaleGestureDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            private var lastToggleTime = 0L
            private val COOLDOWN = 500L

            override fun onScale(detector: ScaleGestureDetector): Boolean {
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastToggleTime < COOLDOWN) return false

                val scaleFactor = detector.scaleFactor
                if (scaleFactor > 1.05f) {
                    // Pinch out -> Zoom in to 1.0x
                    cameraWebStreamServer?.setZoomRatio(1.0f)
                    lastToggleTime = currentTime
                    return true
                } else if (scaleFactor < 0.95f) {
                    // Pinch in -> Zoom out to Wide (e.g. 0.5x)
                    val minZoom = cameraWebStreamServer?.getMinZoomRatio() ?: 1.0f
                    cameraWebStreamServer?.setZoomRatio(minZoom)
                    lastToggleTime = currentTime
                    return true
                }
                return false
            }
        })

        cameraOverlay.setOnTouchListener { v, event ->
            scaleGestureDetector.onTouchEvent(event)
            if (event.pointerCount >= 2) {
                true 
            } else {
                if (event.action == android.view.MotionEvent.ACTION_UP) {
                    v.performClick()
                }
                false
            }
        }
    }

    private fun setupSeekBarListeners() {
        val listener = object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (isAutoMode && fromUser) return
                updateValueLabels()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        }
        speedSeekBar.setOnSeekBarChangeListener(listener)
        rpmSeekBar.setOnSeekBarChangeListener(listener)
        accelSeekBar.setOnSeekBarChangeListener(listener)
    }

    private fun updateValueLabels() {
        if (!isAutoMode) {
            speedValueText.text = "${speedSeekBar.progress} km/h"
            rpmValueText.text = "${rpmSeekBar.progress}"
            val accel = accelSeekBar.progress / 10.0f
            accelValueText.text = String.format("%.2f", accel)
            
            speedValueText.setTextColor(Color.parseColor("#00ff88"))
            rpmValueText.setTextColor(Color.parseColor("#00ff88"))
            accelValueText.setTextColor(Color.parseColor("#00ff88"))
        }
    }

    private fun animatePill(auto: Boolean) {
        val container = findViewById<View>(R.id.modeSwitchContainer)
        val targetX = if (auto) 0f else (container.width / 2f)
        ObjectAnimator.ofFloat(pillSelector, "translationX", targetX).apply {
            duration = 300
            interpolator = AccelerateDecelerateInterpolator()
            start()
        }
        
        textAuto.setTextColor(if (auto) Color.BLACK else Color.WHITE)
        textManual.setTextColor(if (auto) Color.WHITE else Color.BLACK)
    }

    private fun updateUiMode() {
        speedSeekBar.isEnabled = !isAutoMode
        rpmSeekBar.isEnabled = !isAutoMode
        accelSeekBar.isEnabled = !isAutoMode

        if (isAutoMode) {
            startAutoSensors()
            speedSeekBar.progress = 0
            speedValueText.text = "0 km/h"
            accelSeekBar.progress = 0
            accelValueText.text = "0.00"
            rpmValueText.text = "null"
            rpmValueText.setTextColor(Color.parseColor("#888888"))
            rpmSeekBar.progress = 0
            currentGpsSpeed = 0f
            currentLinearAccel = 0f
        } else {
            stopAutoSensors()
            updateValueLabels()
        }
    }

    private fun checkPermissions() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                toggleStreaming(true)
            }
        }
    }

    private fun startAutoSensors() {
        linearAccelSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 0L, 0f, this)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun stopAutoSensors() {
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (isAutoMode && event?.sensor?.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]
            currentLinearAccel = Math.sqrt((x * x + y * y + z * z).toDouble()).toFloat()
            accelSeekBar.progress = (currentLinearAccel!! * 10).toInt().coerceIn(0, 200)
            accelValueText.text = String.format("%.2f", currentLinearAccel)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        if (isAutoMode) {
            currentGpsSpeed = location.speed * 3.6f
            speedSeekBar.progress = currentGpsSpeed!!.toInt().coerceIn(0, 250)
            speedValueText.text = "${speedSeekBar.progress} km/h"
        }
    }

    private fun broadcastTelemetry() {
        val json = JSONObject().apply {
            if (isAutoMode) {
                put("speed_kmh", if (currentGpsSpeed != null) currentGpsSpeed!!.toInt() else JSONObject.NULL)
                put("accel", if (currentLinearAccel != null) currentLinearAccel else JSONObject.NULL)
                put("rpm", JSONObject.NULL)
            } else {
                put("speed_kmh", speedSeekBar.progress)
                put("rpm", rpmSeekBar.progress)
                put("accel", accelSeekBar.progress / 10.0)
            }
            put("type", "telemetry") // Discriminate from binary frames on receiver if needed
        }

        sendTelemetryHttp(json.toString())
    }

    private fun sendTelemetryHttp(payload: String) {
        if (telemetryRequestInFlight) {
            return
        }

        telemetryRequestInFlight = true

        val endpoint = telemetryEndpoint()
        val body = payload.toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url(endpoint)
            .post(body)
            .build()

        telemetryHttpClient.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                telemetryRequestInFlight = false
                telemetryOnline = false
                runOnUiThread { updateStatusDisplay() }
            }

            override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                telemetryRequestInFlight = false
                telemetryOnline = response.isSuccessful
                response.close()
                runOnUiThread { updateStatusDisplay() }
            }
        })
    }

    private fun telemetryEndpoint(): String {
        val prefs = getSharedPreferences("TelemetryPrefs", MODE_PRIVATE)
        val ip = prefs.getString("server_ip", "192.168.1.100")?.trim().orEmpty()
        val port = prefs.getString("server_port", "5500")?.trim().orEmpty()
        return "http://$ip:$port/telemetry"
    }

    private fun updateStatusDisplay() {
        val ipStr = getLocalIpAddress() ?: "127.0.0.1"
        val status = if (telemetryOnline) "Online" else "Connecting..."
        val streamUrl = if (isStreaming) {
            cameraWebStreamServer?.getStreamUrl(ipStr)
        } else {
            null
        }

        statusText.text = if (streamUrl != null) {
            "Status: $status | IP: $ipStr | Stream: $streamUrl"
        } else {
            "Status: $status | IP: $ipStr"
        }
    }

    private fun setStatusText(message: String) {
        statusText.text = "Status: $message"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraWebStreamServer?.stop()
        handler.removeCallbacks(sendTelemetryRunnable)
        stopAutoSensors()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
