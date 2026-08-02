# RagbaarNet Platform App Implementation

This app serves as a thin Android wrapper for the RagbaarNet Platform UI.

## Architecture
The application is built around a full-screen **WebView** (`MainActivity.kt`) that connects to the central RagbaarNet processor server.

### Key Components
- **WebView Interface**: Configured with JavaScript enabled, DOM storage, and media playback support without user gestures.
- **Permission Handling**: Uses `WebChromeClient.onPermissionRequest` to grant WebRTC permissions (Camera and Audio) to the server-hosted UI.
- **Navigation**: Implements a two-finger long-press gesture to reveal the URL input bar for debugging or changing server endpoints.
- **Last URL Persistence**: Stores the last successful connection in `SharedPreferences` for automatic reconnection.

## Protocols Used
- **HTTP/HTTPS**: Used for fetching the HTML UI, CSS, and JavaScript assets from the platform server (default port 5000).
- **WebRTC**: Enabled via the `WebView` to support real-time audio/video interaction between the browser-based UI and the hardware.
