use sqlx::SqlitePool;

use crate::database::models::{Recording, TranscriptSegment};

#[derive(Debug)]
pub struct RecordingWithSegments {
    pub recording: Recording,
    pub segments: Vec<TranscriptSegment>,
}

pub struct RecordingsRepository;

impl RecordingsRepository {
    pub async fn get_recordings(pool: &SqlitePool) -> Result<Vec<Recording>, sqlx::Error> {
        let rows = sqlx::query_as::<_, Recording>(
            "SELECT id,title,created_at,updated_at,duration_ms,audio_path,folder_path,source,asr_engine,language,status FROM recordings ORDER BY created_at DESC"
        )
        .fetch_all(pool)
        .await?;
        Ok(rows)
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

    pub async fn delete_recording(pool: &SqlitePool, id: &str) -> Result<bool, sqlx::Error> {
        let result = sqlx::query("DELETE FROM recordings WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}
