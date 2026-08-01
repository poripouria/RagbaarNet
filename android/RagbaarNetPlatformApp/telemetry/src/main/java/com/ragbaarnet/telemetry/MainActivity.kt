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
    private lateinit var streamToggleButton: MaterialButton
    
    private lateinit var pillSelector: View
    private lateinit var textAuto: TextView
    private lateinit var textManual: TextView

    private var isAutoMode = true
    private var isStreaming = false
    private val handler = Handler(Looper.getMainLooper())
    private var webSocketManager: WebSocketManager? = null
    private var cameraWebStreamServer: CameraWebStreamServer? = null

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
        streamToggleButton = findViewById(R.id.streamToggleButton)
        
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
            
            cameraOverlay.visibility = View.VISIBLE
            
            try {
                if (cameraWebStreamServer == null) {
                    cameraWebStreamServer = CameraWebStreamServer(this) { message ->
                        runOnUiThread {
                            setStatusText(message)
                            updateStatusDisplay()
                        }
                    }
                    cameraWebStreamServer?.start()
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
                cameraWebStreamServer?.stop()
                cameraWebStreamServer = null
                setStatusText("Stream failed to start: ${startErr.message ?: "unknown error"}")
                updateStatusDisplay()
                Toast.makeText(this, "Stream failed to start", Toast.LENGTH_SHORT).show()
            }
        } else {
            isStreaming = false
            streamToggleButton.text = "Stream: OFF"
            streamToggleButton.setTextColor(Color.WHITE)
            streamToggleButton.setBackgroundColor(Color.TRANSPARENT)
            streamToggleButton.setStrokeColor(ContextCompat.getColorStateList(this, android.R.color.white))
            
            cameraOverlay.visibility = View.GONE
            
            cameraWebStreamServer?.stop()
            cameraWebStreamServer = null

            updateStatusDisplay()
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
        val ipStr = getLocalIpAddress() ?: "127.0.0.1"
        val status = if (webSocketManager?.isConnected() == true) "Online" else "Connecting..."
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
        webSocketManager?.disconnect()
        cameraWebStreamServer?.stop()
        handler.removeCallbacks(sendTelemetryRunnable)
        stopAutoSensors()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}
