use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Recording {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub duration_ms: Option<i64>,
    pub audio_path: Option<String>,
    pub folder_path: Option<String>,
    pub source: Option<String>,
    pub asr_engine: Option<String>,
    pub language: Option<String>,
    pub status: Option<String>,
}
