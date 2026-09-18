use log::info;
use tauri::{AppHandle, Emitter, Manager};

use crate::state::AppState;

/// Return true only when the SQLite database file does not yet exist.
///
/// The current startup path normally initializes the database before the
/// frontend can invoke this command, so established installs will return false.
#[tauri::command]
pub async fn check_first_launch(_app: AppHandle) -> Result<bool, String> {
    Ok(!super::setup::database_path().exists())
}

/// Ensure a fresh database is initialized.
///
/// VoxMinutes currently initializes SQLite during Tauri setup. This command is
/// retained for the frontend onboarding contract and initializes the database
/// only if AppState is not already managed.
#[tauri::command]
pub async fn initialize_fresh_database(app: AppHandle) -> Result<(), String> {
    if app.try_state::<AppState>().is_none() {
        super::setup::initialize_database_on_startup(&app).await?;
    }

    app.emit("database-initialized", ())
        .map_err(|e| format!("Failed to emit database-initialized event: {}", e))?;

    Ok(())
}

/// Get the directory containing the VoxMinutes SQLite database.
#[tauri::command]
pub async fn get_database_directory(_app: AppHandle) -> Result<String, String> {
    let db_path = super::setup::database_path();
    let dir = db_path
        .parent()
        .ok_or_else(|| "Database path has no parent directory".to_string())?;

    Ok(dir.to_string_lossy().to_string())
}

/// Open the database directory in the system file explorer.
#[tauri::command]
pub async fn open_database_folder(_app: AppHandle) -> Result<(), String> {
    let db_path = super::setup::database_path();
    let dir = db_path
        .parent()
        .ok_or_else(|| "Database path has no parent directory".to_string())?;

    std::fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create database directory: {}", e))?;

    let folder_path = dir.to_string_lossy().to_string();

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open database folder: {}", e))?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open database folder: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&folder_path)
            .spawn()
            .map_err(|e| format!("Failed to open database folder: {}", e))?;
    }

    info!("Opened database folder: {}", folder_path);
    Ok(())
}
