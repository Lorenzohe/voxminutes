use sqlx::{sqlite::SqlitePool, SqlitePool as Pool};

#[derive(Clone)]
pub struct DatabaseManager {
    pool: Pool,
}

impl DatabaseManager {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
