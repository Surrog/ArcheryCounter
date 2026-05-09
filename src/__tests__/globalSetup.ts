/**
 * Jest globalSetup: runs once before all test suites.
 *
 * Creates the per-test-file schemas so each test suite operates on its own
 * isolated tables and concurrent runs cannot interfere:
 *
 *   test_td      — used by targetDetection.test.ts (generated table)
 *   test_scripts — used by scripts.test.ts + annotate subprocesses
 *                  (generated + annotations tables)
 */
import { Pool } from 'pg';

const GENERATED_DDL = `
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
`;

const ANNOTATIONS_DDL = `
  CREATE TABLE IF NOT EXISTS annotations (
    filename       TEXT PRIMARY KEY,
    paper_boundary JSONB,
    rings          JSONB NOT NULL DEFAULT '[]',
    arrows         JSONB NOT NULL DEFAULT '[]',
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
`;

export default async function globalSetup(): Promise<void> {
  const baseConfig = {
    host:     process.env.DB_HOST     || 'localhost',
    port:     parseInt(process.env.DB_PORT || '5432'),
    user:     process.env.DB_USER     || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    database: process.env.DB_NAME     || 'postgres',
  };

  for (const schema of ['test_td', 'test_scripts']) {
    const db = new Pool(baseConfig);
    try {
      await db.query(`CREATE SCHEMA IF NOT EXISTS "${schema}"`);
      // Use fully-qualified names so we don't depend on search_path.
      await db.query(GENERATED_DDL.replace(/\bgenerated\b/g, `"${schema}".generated`));
      if (schema === 'test_scripts') {
        await db.query(ANNOTATIONS_DDL.replace(/\bannotations\b/g, `"${schema}".annotations`));
      }
    } finally {
      await db.end();
    }
  }
}
