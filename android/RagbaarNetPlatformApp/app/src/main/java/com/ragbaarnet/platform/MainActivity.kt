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
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

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

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                return false
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: android.webkit.WebResourceError?
            ) {
                // Show a friendly error or just log it
                if (request?.isForMainFrame == true) {
                    val errorHtml = "<html><body><div style='padding:20px; font-family:sans-serif;'>" +
                            "<h2>Connection Error</h2>" +
                            "<p>Cannot reach server at ${BuildConfig.SERVER_URL}</p>" +
                            "<p>Error: ${error?.description}</p>" +
                            "<p><b>Tips:</b><br>1. Check if the server is running.<br>" +
                            "2. Ensure ADB Reverse is active (if using USB).<br>" +
                            "3. Check Wi-Fi connection.</p>" +
                            "</div></body></html>"
                    view?.loadData(errorHtml, "text/html", "UTF-8")
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                // Grant WebRTC permissions (camera/mic) for the loaded page.
                request.grant(request.resources)
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

        // Load the Platform UI from the processor server running on the laptop.
        webView.loadUrl(BuildConfig.SERVER_URL)
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
