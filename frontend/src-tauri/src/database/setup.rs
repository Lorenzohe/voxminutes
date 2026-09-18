use std::path::PathBuf;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use tauri::{AppHandle, Manager};

use super::manager::DatabaseManager;
use crate::state::AppState;

pub(crate) fn database_path() -> PathBuf {
    let base = dirs::data_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let dir = base.join("VoxMinutes");
    std::fs::create_dir_all(&dir).expect("failed to create database directory");
    dir.join("voxminutes.db")
}

async fn run_migrations(pool: &sqlx::SqlitePool) -> Result<(), String> {
    let schema = include_str!("../../migrations/20260717000000_mvp_initial_schema.sql");

    for statement in schema.split(';') {
        let statement = statement.trim();
        if !statement.is_empty() {
            sqlx::query(statement)
                .execute(pool)
                .await
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

pub async fn initialize_database_on_startup<R: tauri::Runtime>(
    app: &AppHandle<R>,
) -> Result<(), String> {
    let path = database_path();
    let options = SqliteConnectOptions::new()
        .filename(&path)
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
        .map_err(|e| e.to_string())?;

    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(&pool)
        .await
        .map_err(|e| e.to_string())?;

    run_migrations(&pool).await?;

    let manager = DatabaseManager::new(pool);

    app.manage(AppState { db_manager: manager });

    Ok(())
}
