use sqlx::SqlitePool;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SettingRow {
    pub key: String,
    pub value: String,
}

pub struct SettingsRepository;

impl SettingsRepository {
    pub async fn get(pool: &SqlitePool, key: &str) -> Result<Option<String>, sqlx::Error> {
        sqlx::query_scalar::<_, String>("SELECT value FROM settings WHERE key = ?")
            .bind(key)
            .fetch_optional(pool)
            .await
    }

    pub async fn set(pool: &SqlitePool, key: &str, value: &str) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        .bind(key)
        .bind(value)
        .execute(pool)
        .await?;
        Ok(())
    }

    pub async fn delete(pool: &SqlitePool, key: &str) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM settings WHERE key = ?")
            .bind(key)
            .execute(pool)
            .await?;
        Ok(())
    }

    pub async fn get_all(pool: &SqlitePool) -> Result<Vec<SettingRow>, sqlx::Error> {
        sqlx::query_as::<_, SettingRow>("SELECT key, value FROM settings ORDER BY key")
            .fetch_all(pool)
            .await
    }

    pub async fn get_export_dir(pool: &SqlitePool) -> Result<String, sqlx::Error> {
        Ok(Self::get(pool, "export.default_dir").await?.unwrap_or_default())
    }
}
