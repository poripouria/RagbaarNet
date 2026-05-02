package com.ragbaarnet.platform

import android.Manifest
import android.annotation.SuppressLint
import android.net.Uri
import android.os.Bundle
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private lateinit var urlInput: EditText
    private lateinit var connectButton: Button
    private lateinit var inputContainer: LinearLayout
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

    private val PREFS_NAME = "RagbaarPrefs"
    private val KEY_URL = "last_url"

    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* WebView will re-request as needed */ }

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val callback = fileChooserCallback
        fileChooserCallback = null

        if (callback == null) return@registerForActivityResult

        val uris = WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
        callback.onReceiveValue(uris)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        urlInput = findViewById(R.id.urlInput)
        connectButton = findViewById(R.id.connectButton)
        inputContainer = findViewById(R.id.inputContainer)

        val sharedPrefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val lastUrl = sharedPrefs.getString(KEY_URL, BuildConfig.SERVER_URL)
        urlInput.setText(lastUrl)

        connectButton.setOnClickListener {
            var url = urlInput.text.toString().trim()
            if (url.isNotEmpty()) {
                if (!url.startsWith("http://") && !url.startsWith("https://")) {
                    url = "http://$url"
                }
                
                // Save for next time
                sharedPrefs.edit().putString(KEY_URL, url).apply()
                
                webView.loadUrl(url)
            }
        }

        // Long press on WebView to show the URL bar again
        webView.setOnLongClickListener {
            inputContainer.visibility = View.VISIBLE
            true
        }

        // Runtime permissions for getUserMedia() (camera/mic)
        requestPermissionsLauncher.launch(
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
            )
        )

        val ws: WebSettings = webView.settings
        ws.javaScriptEnabled = true
        ws.domStorageEnabled = true
        ws.allowFileAccess = true
        ws.allowContentAccess = true
        ws.mediaPlaybackRequiresUserGesture = false
        ws.loadWithOverviewMode = true
        ws.useWideViewPort = true
        ws.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        
        // Enable getUserMedia() and getDisplayMedia() support (API 21+)
        ws.mediaPlaybackRequiresUserGesture = false
        
        // Enable audio focus for getUserMedia()
        this.volumeControlStream = android.media.AudioManager.STREAM_VOICE_CALL

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                return false
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                // Hide input bar after successful load
                inputContainer.visibility = View.GONE
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: android.webkit.WebResourceError?
            ) {
                // Show a friendly error or just log it
                if (request?.isForMainFrame == true) {
                    inputContainer.visibility = View.VISIBLE
                    val currentUrl = view?.url ?: ""
                    val errorHtml = "<html><body><div style='padding:20px; font-family:sans-serif;'>" +
                            "<h2>Connection Error</h2>" +
                            "<p>Cannot reach server at $currentUrl</p>" +
                            "<p>Error: ${error?.description}</p>" +
                            "<p><b>Tips:</b><br>1. Check if the server is running.<br>" +
                            "2. Ensure ADB Reverse is active (if using USB).<br>" +
                            "3. Check Wi-Fi connection.<br>" +
                            "4. <b>Long-press anywhere</b> to show the URL bar again.</p>" +
                            "</div></body></html>"
                    view?.loadData(errorHtml, "text/html", "UTF-8")
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                // Grant both WebRTC and Media permissions (camera/mic) for the loaded page.
                // This handles both getUserMedia() and getDisplayMedia() calls.
                android.util.Log.d("WebChromeClient", "Permission request received for resources: ${request.resources.joinToString(",")}")
                
                // Filter to only grant CAMERA and MICROPHONE permissions
                val allowedResources = request.resources.filter { resource ->
                    resource == PermissionRequest.RESOURCE_VIDEO_CAPTURE || 
                    resource == PermissionRequest.RESOURCE_AUDIO_CAPTURE
                }
                
                if (allowedResources.isNotEmpty()) {
                    android.util.Log.d("WebChromeClient", "Granting permissions: ${allowedResources.joinToString(",")}")
                    request.grant(allowedResources.toTypedArray())
                } else {
                    android.util.Log.w("WebChromeClient", "No supported resources in permission request")
                    request.deny()
                }
            }

            override fun onShowFileChooser(
                webView: WebView,
                filePathCallback: ValueCallback<Array<Uri>>,
                fileChooserParams: FileChooserParams
            ): Boolean {
                // Cancel any pending chooser
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback

                return try {
                    val intent = fileChooserParams.createIntent()
                    fileChooserLauncher.launch(intent)
                    true
                } catch (e: Exception) {
                    fileChooserCallback = null
                    false
                }
            }
        }

        // Automatically load the last used URL
        if (lastUrl != null && lastUrl.isNotEmpty()) {
            webView.loadUrl(lastUrl)
        }
    }

    override fun onDestroy() {
        // Avoid leaking the WebView
        try {
            webView.destroy()
        } catch (_: Exception) {
        }
        super.onDestroy()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (this::webView.isInitialized && webView.canGoBack()) {
            webView.goBack()
            return
        }
        super.onBackPressed()
    }
}
