package com.ragbaarnet.telemetry

import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.textfield.TextInputEditText
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.IOException

class SettingsActivity : AppCompatActivity() {

    private lateinit var ipInput: TextInputEditText
    private lateinit var portInput: TextInputEditText
    private lateinit var testButton: Button
    private lateinit var saveButton: Button
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        ipInput = findViewById(R.id.ipInput)
        portInput = findViewById(R.id.portInput)
        testButton = findViewById(R.id.testButton)
        saveButton = findViewById(R.id.saveButton)

        val prefs = getSharedPreferences("TelemetryPrefs", MODE_PRIVATE)
        ipInput.setText(prefs.getString("server_ip", "192.168.1.100"))
        portInput.setText(prefs.getString("server_port", "5500"))

        testButton.setOnClickListener {
            testConnection()
        }

        saveButton.setOnClickListener {
            val ip = ipInput.text.toString().trim()
            val port = portInput.text.toString().trim()
            if (ip.isNotEmpty() && port.isNotEmpty()) {
                prefs.edit().apply {
                    putString("server_ip", ip)
                    putString("server_port", port)
                    apply()
                }
                Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun testConnection() {
        val ip = ipInput.text.toString().trim()
        val port = portInput.text.toString().trim()
        val url = "http://$ip:$port/telemetry"

        val request = Request.Builder().url(url).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@SettingsActivity, "Connection failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                runOnUiThread {
                    if (response.code == 404 || response.code == 405) {
                        // 405 is likely because we did GET instead of POST, but at least the server is there
                        Toast.makeText(this@SettingsActivity, "Server reachable!", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@SettingsActivity, "Server responded with ${response.code}", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }
}
