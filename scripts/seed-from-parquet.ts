import * as path from 'path';
import { Pool } from 'pg';
import { ParquetReader } from '@dsnp/parquetjs';

const PARQUET_PATH = path.resolve(__dirname, '../data/annotations.parquet');

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

async function main() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS annotations (
      filename       TEXT PRIMARY KEY,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);

  const reader = await ParquetReader.openFile(PARQUET_PATH);
  const cursor = reader.getCursor();

  let count = 0;
  let record: Record<string, unknown>;
  while ((record = await cursor.next())) {
    await db.query(
      `INSERT INTO annotations (filename, paper_boundary, rings)
       VALUES ($1, $2, $3)
       ON CONFLICT (filename) DO NOTHING`,
      [
        record.filename,
        record.paper_boundary as string,
        record.rings as string,
      ],
    );
    count++;
  }
  await reader.close();
  console.log(`Seeded ${count} rows from ${PARQUET_PATH}`);
}

main().catch(err => { console.error(err); process.exit(1); })
  .finally(() => db.end());
