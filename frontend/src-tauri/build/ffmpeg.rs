use std::path::Path;

/// Ensure FFmpeg resources are available during build.
///
/// The application can use bundled FFmpeg resources when present.
/// This build step intentionally does not fail when FFmpeg is managed
/// externally, allowing CI builds to proceed.
pub fn ensure_ffmpeg_binary() {
    let ffmpeg_path = Path::new("binaries/ffmpeg");

    if ffmpeg_path.exists() {
        println!("cargo:warning=FFmpeg binary found: {:?}", ffmpeg_path);
    } else {
        println!("cargo:warning=FFmpeg binary not found at {:?}; continuing build", ffmpeg_path);
    }
}
