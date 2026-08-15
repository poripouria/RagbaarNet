# RagbaarNet Telemetry App Implementation

The Telemetry app is a standalone tool for collecting vehicle dynamics and streaming real-time video from an Android device.

## Features
- **Sensor Fusion**: Captures linear acceleration and GPS speed.
- **Auto/Manual Modes**: Supports both real sensor data and manual overrides via SeekBars.
- **Video Streaming**: MJPEG over HTTP server for low-latency visual feedback.

## Architecture
- **CameraWebStreamServer**: A custom `ServerSocket` implementation that handles multiple clients. It captures frames using `ImageReader` (YUV_420_888), converts them to JPEG, and broadcasts them as an MJPEG stream (`multipart/x-mixed-replace`).
- **Telemetry Broadcast**: A background `Runnable` sends sensor data to the processor server every 250ms via HTTP POST.
- **Dynamic Zoom**: Supports switching between standard and wide-angle lenses (if available) via `CONTROL_ZOOM_RATIO`.

## Protocols Used
- **HTTP (JSON)**: Telemetry data is posted as JSON to `http://<server>:<port>/telemetry`.
- **MJPEG over HTTP**: Video is served at `http://<device_ip>:8080/stream.webm` using standard MJPEG multipart headers.
