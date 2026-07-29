package com.ragbaarnet.telemetry

import android.Manifest
import android.animation.ObjectAnimator
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
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
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.ImageButton
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    private lateinit var speedSeekBar: SeekBar
    private lateinit var rpmSeekBar: SeekBar
    private lateinit var accelSeekBar: SeekBar
    private lateinit var speedValueText: TextView
    private lateinit var rpmValueText: TextView
    private lateinit var accelValueText: TextView
    private lateinit var statusText: TextView
    private lateinit var viewFinder: PreviewView
    private lateinit var cameraOverlay: View
    private lateinit var streamToggleButton: MaterialButton
    
    private lateinit var pillSelector: View
    private lateinit var textAuto: TextView
    private lateinit var textManual: TextView

    private var isAutoMode = true
    private var isStreaming = false
    private val handler = Handler(Looper.getMainLooper())
    private var cameraExecutor: ExecutorService? = null
    private var webSocketManager: WebSocketManager? = null
    private var cameraProvider: ProcessCameraProvider? = null

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
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        speedSeekBar = findViewById(R.id.speedSeekBar)
        rpmSeekBar = findViewById(R.id.rpmSeekBar)
        accelSeekBar = findViewById(R.id.accelSeekBar)
        speedValueText = findViewById(R.id.speedValueText)
        rpmValueText = findViewById(R.id.rpmValueText)
        accelValueText = findViewById(R.id.accelValueText)
        statusText = findViewById(R.id.statusText)
        viewFinder = findViewById(R.id.viewFinder)
        cameraOverlay = findViewById(R.id.cameraOverlay)
        streamToggleButton = findViewById(R.id.streamToggleButton)
        
        pillSelector = findViewById(R.id.pillSelector)
        textAuto = findViewById(R.id.textAuto)
        textManual = findViewById(R.id.textManual)

        findViewById<ImageButton>(R.id.settingsButton).setOnClickListener {
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

        setupSeekBarListeners()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        linearAccelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        updateUiMode()
        checkPermissions()
        initWebSocket()
        handler.post(sendTelemetryRunnable)
    }

    private fun initWebSocket() {
        val prefs = getSharedPreferences("TelemetryPrefs", MODE_PRIVATE)
        val ip = prefs.getString("server_ip", "192.168.1.100")
        val port = prefs.getString("server_port", "5500")
        // Note: The platform server will need to handle /ws endpoint or similar
        val url = "ws://$ip:$port/telemetry" 

        webSocketManager = WebSocketManager(url).apply {
            listener = object : WebSocketManager.ConnectionListener {
                override fun onConnected() {
                    runOnUiThread { updateStatusDisplay() }
                }
                override fun onDisconnected() {
                    runOnUiThread { updateStatusDisplay() }
                }
                override fun onError(error: String) {
                    runOnUiThread { updateStatusDisplay() }
                }
            }
            connect()
        }
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
            
            viewFinder.visibility = View.VISIBLE
            cameraOverlay.visibility = View.VISIBLE
            
            cameraExecutor = Executors.newSingleThreadExecutor()
            startCamera()
            
            Toast.makeText(this, "Camera stream over WebSocket active", Toast.LENGTH_SHORT).show()
        } else {
            isStreaming = false
            streamToggleButton.text = "Stream: OFF"
            streamToggleButton.setTextColor(Color.WHITE)
            streamToggleButton.setBackgroundColor(Color.TRANSPARENT)
            streamToggleButton.setStrokeColor(ContextCompat.getColorStateList(this, android.R.color.white))
            
            viewFinder.visibility = View.GONE
            cameraOverlay.visibility = View.GONE
            
            stopCamera()
            cameraExecutor?.shutdown()
            cameraExecutor = null
        }
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

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                        // CameraX pipeline: ImageProxy -> NV21 -> JPEG
                        val yuvImage = YuvImage(
                            yuvBytes(imageProxy),
                            ImageFormat.NV21,
                            imageProxy.width,
                            imageProxy.height,
                            null
                        )
                        val out = ByteArrayOutputStream()
                        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 70, out)
                        
                        // TRANSMIT BINARY JPEG via WebSocket
                        webSocketManager?.sendFrame(out.toByteArray())
                        
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("MainActivity", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
    }

    private fun yuvBytes(image: androidx.camera.core.ImageProxy): ByteArray {
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

        return nv21
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

        // TRANSMIT JSON TELEMETRY via WebSocket
        webSocketManager?.sendTelemetry(json.toString())
    }

    private fun updateStatusDisplay() {
        val ipStr = getLocalIpAddress() ?: "Unknown"
        val status = if (webSocketManager?.isConnected() == true) "Online" else "Connecting..."
        statusText.text = "Status: $status | IP: $ipStr"
    }

    override fun onDestroy() {
        super.onDestroy()
        webSocketManager?.disconnect()
        cameraExecutor?.shutdown()
        handler.removeCallbacks(sendTelemetryRunnable)
        stopAutoSensors()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
