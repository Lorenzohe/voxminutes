/// FFmpeg is resolved by the application at runtime.
///
/// Runtime lookup checks for a bundled binary first, then PATH, and finally
/// uses ffmpeg-sidecar to download FFmpeg when needed. Keeping FFmpeg out of
/// Tauri's externalBin list avoids requiring a platform-specific binary to be
/// present in the source tree just to compile the application.
pub fn ensure_ffmpeg_binary() {
    println!(
        "cargo:warning=FFmpeg is resolved at runtime; no build-time bundled binary is required"
    );
}
