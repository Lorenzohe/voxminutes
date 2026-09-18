use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

use crate::database::models::{Recording, TranscriptSegment};

#[derive(Debug)]
pub struct RecordingWithSegments {
    pub recording: Recording,
    pub segments: Vec<TranscriptSegment>,
}

pub struct RecordingsRepository;

impl RecordingsRepository {
    pub async fn get_recordings(pool: &SqlitePool) -> Result<Vec<Recording>, sqlx::Error> {
        sqlx::query_as::<_, Recording>(
            "SELECT id,title,created_at,updated_at,duration_ms,audio_path,folder_path,source,asr_engine,language,status FROM recordings ORDER BY created_at DESC"
        )
        .fetch_all(pool)
        .await
    }

    pub async fn get_recording(
        pool: &SqlitePool,
        id: &str,
    ) -> Result<Option<RecordingWithSegments>, sqlx::Error> {
        let recording = sqlx::query_as::<_, Recording>(
            "SELECT id,title,created_at,updated_at,duration_ms,audio_path,folder_path,source,asr_engine,language,status FROM recordings WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;

        let Some(recording) = recording else {
            return Ok(None);
        };

        let segments = sqlx::query_as::<_, TranscriptSegment>(
            "SELECT id,recording_id,text,start_ms,end_ms,speaker,source,created_at FROM transcript_segments WHERE recording_id = ? ORDER BY start_ms ASC"
        )
        .bind(id)
        .fetch_all(pool)
        .await?;

        Ok(Some(RecordingWithSegments { recording, segments }))
    }

    pub async fn create_recording(
        pool: &SqlitePool,
        title: &str,
        audio_path: Option<&str>,
        source: Option<&str>,
        asr_engine: Option<&str>,
        language: Option<&str>,
        folder_path: Option<&str>,
    ) -> Result<String, sqlx::Error> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();

        sqlx::query(
            "INSERT INTO recordings (id,title,created_at,updated_at,duration_ms,audio_path,folder_path,source,asr_engine,language,status) VALUES (?,?,?,?,0,?,?,?,?,?,'completed')"
        )
        .bind(&id)
        .bind(title)
        .bind(now)
        .bind(now)
        .bind(audio_path)
        .bind(folder_path)
        .bind(source.unwrap_or("realtime"))
        .bind(asr_engine)
        .bind(language)
        .execute(pool)
        .await?;

        Ok(id)
    }

    pub async fn update_recording_title(
        pool: &SqlitePool,
        id: &str,
        title: &str,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query("UPDATE recordings SET title = ?, updated_at = ? WHERE id = ?")
            .bind(title)
            .bind(Utc::now())
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn update_recording_duration(
        pool: &SqlitePool,
        id: &str,
        duration_ms: i64,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query(
            "UPDATE recordings SET duration_ms = ?, updated_at = ? WHERE id = ?"
        )
        .bind(duration_ms)
        .bind(Utc::now())
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn delete_recording(pool: &SqlitePool, id: &str) -> Result<bool, sqlx::Error> {
        let result = sqlx::query("DELETE FROM recordings WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}
