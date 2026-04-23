/**
 * Jest globalSetup: runs once before all test suites.
 *
 * Populates the `generated` table with detection results for all images in
 * images/.  Individual test suites then read from this table (fast DB query)
 * rather than running the detection algorithm inline.
 */
import { Pool } from 'pg';

export default async function globalSetup(): Promise<void> {
  const db = new Pool({
    host:     process.env.DB_HOST     || 'localhost',
    port:     parseInt(process.env.DB_PORT || '5432'),
    user:     process.env.DB_USER     || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    database: process.env.DB_NAME     || 'postgres',
  });

  try {
    await db.query(`
      CREATE TABLE IF NOT EXISTS generated (
        filename       TEXT PRIMARY KEY,
        algorithm_hash TEXT NOT NULL,
        paper_boundary JSONB,
        rings          JSONB NOT NULL DEFAULT '[]',
        arrows         JSONB NOT NULL DEFAULT '[]',
        width          INT,
        height         INT,
        updated_at     TIMESTAMPTZ DEFAULT NOW()
      )
    `);
  } finally {
    await db.end();
  }
}
