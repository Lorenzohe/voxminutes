use std::path::PathBuf;

use sqlx::sqlite::SqlitePoolOptions;
use tauri::{AppHandle, Manager};

use super::manager::DatabaseManager;
use crate::state::AppState;

fn database_path() -> PathBuf {
    let base = dirs::data_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let dir = base.join("VoxMinutes");
    std::fs::create_dir_all(&dir).expect("failed to create database directory");
    dir.join("voxminutes.db")
}

pub async fn initialize_database_on_startup<R: tauri::Runtime>(
    app: &AppHandle<R>,
) -> Result<(), String> {
    let path = database_path();
    let url = format!("sqlite:{}", path.to_string_lossy());

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .map_err(|e| e.to_string())?;

    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&pool)
        .await
        .map_err(|e| e.to_string())?;

    let manager = DatabaseManager::new(pool);

    app.manage(AppState { db_manager: manager });

    Ok(())
}
