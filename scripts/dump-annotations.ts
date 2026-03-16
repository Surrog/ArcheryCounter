import * as path from 'path';
import { Pool } from 'pg';
import { ParquetSchema, ParquetWriter } from '@dsnp/parquetjs';

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
    'SELECT filename, paper_boundary, rings FROM annotations ORDER BY filename',
  );

  const schema = new ParquetSchema({
    filename:       { type: 'UTF8' },
    paper_boundary: { type: 'UTF8' },
    rings:          { type: 'UTF8' },
  });

  const writer = await ParquetWriter.openFile(schema, OUT_PATH);
  for (const row of rows) {
    await writer.appendRow({
      filename:       row.filename,
      paper_boundary: JSON.stringify(row.paper_boundary ?? null),
      rings:          JSON.stringify(row.rings ?? []),
    });
  }
  await writer.close();
  console.log(`Wrote ${rows.length} rows → ${OUT_PATH}`);
}

main().catch(err => { console.error(err); process.exit(1); })
  .finally(() => db.end());
