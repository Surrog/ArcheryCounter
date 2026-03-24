import * as path from 'path';
import { Pool } from 'pg';
import { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects } from 'hyparquet';

const PARQUET_PATH = path.resolve(import.meta.dirname, '../data/annotations.parquet');

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
      arrows         JSONB NOT NULL DEFAULT '[]',
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
  await db.query(`ALTER TABLE annotations ADD COLUMN IF NOT EXISTS arrows JSONB NOT NULL DEFAULT '[]'`);

  const file = await asyncBufferFromFile(PARQUET_PATH);
  const metadata = await parquetMetadataAsync(file);
  const records = await parquetReadObjects({ file, metadata });

  let count = 0;
  for (const record of records) {
    await db.query(
      `INSERT INTO annotations (filename, paper_boundary, rings, arrows)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (filename) DO UPDATE
         SET paper_boundary = EXCLUDED.paper_boundary,
             rings          = EXCLUDED.rings,
             arrows         = EXCLUDED.arrows,
             updated_at     = NOW()`,
      [
        record.filename,
        record.paper_boundary as string,
        record.rings as string,
        record.arrows as string,
      ],
    );
    count++;
  }
  console.log(`Seeded ${count} rows from ${PARQUET_PATH}`);
}

main().catch(err => { console.error(err); process.exit(1); })
  .finally(() => db.end());
