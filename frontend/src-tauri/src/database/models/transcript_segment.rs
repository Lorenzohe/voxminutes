use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TranscriptSegment {
    pub id: String,
    pub recording_id: String,
    pub text: String,
    pub start_ms: i64,
    pub end_ms: Option<i64>,
    pub speaker: Option<String>,
    pub source: Option<String>,
    pub created_at: DateTime<Utc>,
}
