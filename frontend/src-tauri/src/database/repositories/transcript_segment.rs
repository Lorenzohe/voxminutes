use sqlx::SqlitePool;

use crate::database::models::TranscriptSegment;

pub struct TranscriptSegmentsRepository;

impl TranscriptSegmentsRepository {
    pub async fn get_segments_by_recording(
        pool: &SqlitePool,
        recording_id: &str,
    ) -> Result<Vec<TranscriptSegment>, sqlx::Error> {
        let rows = sqlx::query_as::<_, TranscriptSegment>(
            "SELECT id, recording_id, text, start_ms, end_ms, speaker, source, created_at FROM transcript_segments WHERE recording_id = ? ORDER BY start_ms"
        )
        .bind(recording_id)
        .fetch_all(pool)
        .await?;
        Ok(rows)
    }
}
