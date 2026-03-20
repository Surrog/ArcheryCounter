import * as path from 'path';
import { Pool } from 'pg';
import { parquetWriteFile } from 'hyparquet-writer';

const OUT_PATH = path.resolve(__dirname, '../data/annotations.parquet');

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

async function main() {
  const { rows } = await db.query(
    'SELECT filename, paper_boundary, rings, arrows FROM annotations ORDER BY filename',
  );

  parquetWriteFile({
    filename: OUT_PATH,
    columnData: [
      { name: 'filename',       data: rows.map((r: Record<string, unknown>) => r.filename as string) },
      { name: 'paper_boundary', data: rows.map((r: Record<string, unknown>) => JSON.stringify(r.paper_boundary ?? null)) },
      { name: 'rings',          data: rows.map((r: Record<string, unknown>) => JSON.stringify(r.rings ?? [])) },
      { name: 'arrows',         data: rows.map((r: Record<string, unknown>) => JSON.stringify(r.arrows ?? [])) },
    ],
  });
  console.log(`Wrote ${rows.length} rows → ${OUT_PATH}`);
}

main().catch(err => { console.error(err); process.exit(1); })
  .finally(() => db.end());
