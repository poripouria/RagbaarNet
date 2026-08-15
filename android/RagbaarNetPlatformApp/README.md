# RagbaarNet Platform Android (WebView)

This Android app is a thin WebView wrapper that loads the existing Platform UI served by `modules/Platform/processor.py` at `/ui/`.

## Configure

Edit `android/RagbaarNetPlatformApp/app/build.gradle` and set `BuildConfig.SERVER_URL` to your laptop IP:

Example:

`http://192.168.1.12:5000/ui/`

## Run backend on laptop

From the project root:

`./ragbaarnet-env/Scripts/python.exe main.py `

Then verify from your phone browser (same Wi‑Fi):

`http://<laptop-ip>:5000/ui/`

## Build APK (Android Studio)

- Open `android/RagbaarNetPlatformApp` in Android Studio
- Let Gradle sync/download
- Build > Build Bundle(s) / APK(s) > Build APK(s)
