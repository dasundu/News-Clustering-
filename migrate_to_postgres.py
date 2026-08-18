# migrate_to_postgres.py
"""
One-time migration: copy all data from SQLite (../data/articles.db, kept at
the repo root — not part of the live backend/ code) to Neon PostgreSQL.
Already run; kept only for reference. Run again: python migrate_to_postgres.py
"""

import sqlite3
import psycopg2
import psycopg2.extras
from config import DATABASE_URL

SQLITE_PATH = '../data/articles.db'


def migrate():
    sq = sqlite3.connect(SQLITE_PATH)
    sq.row_factory = sqlite3.Row
    pg = psycopg2.connect(DATABASE_URL)

    with pg.cursor() as cur:
        # ── articles ──────────────────────────────────────────────────────────
        rows = sq.execute('SELECT * FROM articles').fetchall()
        print(f"Migrating {len(rows)} articles...")
        ok = skip = 0
        for r in rows:
            cur.execute('''
                INSERT INTO articles
                    (title, body, url, published_at, source, category,
                     scraped_at, body_fetched, body_cleaned, embedding, cluster_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
            ''', (
                r['title'],
                r['body'],
                r['url'],
                r['published_at'],
                r['source'],
                r['category'],
                r['scraped_at'],
                r['body_fetched'],
                r['body_cleaned'],
                psycopg2.Binary(r['embedding']) if r['embedding'] else None,
                r['cluster_id'],
            ))
            if cur.rowcount > 0:
                ok += 1
            else:
                skip += 1
        print(f"  Articles inserted: {ok}, skipped (duplicate): {skip}")

        # ── custom_sources ────────────────────────────────────────────────────
        try:
            csrows = sq.execute('SELECT * FROM custom_sources').fetchall()
            print(f"Migrating {len(csrows)} custom sources...")
            for r in csrows:
                cur.execute('''
                    INSERT INTO custom_sources (name, url, added_at, last_scraped, active)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                ''', (r['name'], r['url'], r['added_at'], r['last_scraped'], r['active']))
            print(f"  Custom sources done.")
        except Exception as e:
            print(f"  custom_sources table missing or error: {e}")

        # ── cluster_labels ────────────────────────────────────────────────────
        try:
            clrows = sq.execute('SELECT * FROM cluster_labels').fetchall()
            print(f"Migrating {len(clrows)} cluster labels...")
            for r in clrows:
                cur.execute('''
                    INSERT INTO cluster_labels (cluster_id, event_title, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cluster_id) DO NOTHING
                ''', (r['cluster_id'], r['event_title'], r['updated_at']))
            print(f"  Cluster labels done.")
        except Exception as e:
            print(f"  cluster_labels table missing or error: {e}")

    pg.commit()
    pg.close()
    sq.close()
    print("\nMigration complete.")


if __name__ == '__main__':
    migrate()
