# database.py
"""
Database operations for news articles — PostgreSQL (Neon) backend.
"""

import logging
import time
import psycopg2
import psycopg2.extras
import psycopg2.pool
from typing import List, Dict, Tuple, Optional
from config import DATABASE_URL

logger = logging.getLogger(__name__)

_pool = psycopg2.pool.ThreadedConnectionPool(1, 20, DATABASE_URL)
_pool_lock = __import__('threading').Lock()


def _discard(pool, conn):
    """Return a connection to its pool *closed*. Plain putconn() would hand a
    known-broken connection straight back to the next caller."""
    try:
        pool.putconn(conn, close=True)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def _rebuild_pool(stale_pool):
    """
    Replace the global pool, but only if nobody else already has.

    Deliberately does NOT call closeall() on the old pool. closeall() shuts
    every connection it owns — including ones other threads are mid-query on —
    which is exactly how a single stale connection used to take out every
    concurrent request with "cursor already closed" / "connection pointer is
    NULL". The old pool is simply dropped; its connections close when their
    holders release them or when it is garbage collected.
    """
    global _pool
    with _pool_lock:
        if _pool is not stale_pool:
            return                      # another thread already rebuilt it
        last_err = None
        for attempt in range(3):
            try:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 20, DATABASE_URL)
                logger.warning("Rebuilt the database connection pool")
                return
            except Exception as e:      # Neon may still be waking up
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise last_err


def _get_conn():
    """
    A live connection from the pool.

    Neon suspends idle databases, so pooled connections go stale and the first
    use after a quiet period fails. The recovery is deliberately graded:
    a stale connection is discarded *individually* and retried, and the whole
    pool is only replaced when the pool object itself is unusable. That
    ordering matters — the API serves requests from a thread pool, so tearing
    down shared state to recover one connection breaks every request in flight.
    """
    last_err = None
    for _ in range(3):
        pool = _pool                    # snapshot: another thread may swap it
        try:
            conn = pool.getconn()
        # PoolError ("pool is closed"/"exhausted") is a *sibling* of
        # OperationalError, not a subclass, so it must be caught separately —
        # otherwise a pool broken once raises uncaught on every call forever,
        # since nothing else would ever rebuild it. That previously took the
        # scraper down for hours, with each run logged as "executed
        # successfully" because scheduler.py catches per-run exceptions.
        except psycopg2.pool.PoolError as e:
            last_err = e
            _rebuild_pool(pool)
            continue

        if conn.closed:
            _discard(pool, conn)
            last_err = psycopg2.OperationalError("connection was already closed")
            continue

        try:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')  # verify the socket is actually alive
            return conn
        except psycopg2.Error as e:
            last_err = e
            _discard(pool, conn)
            continue

    raise last_err if last_err else psycopg2.OperationalError("could not get a connection")


def _put_conn(conn):
    """
    Release a connection. Tolerates the pool having been swapped underneath:
    putconn() rejects a connection it does not own, and silently dropping it
    would leak the socket, so it is closed instead.
    """
    try:
        _pool.putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


class ArticleDatabase:

    def __init__(self):
        self.init_db()

    def _connect(self):
        return _get_conn()

    def _release(self, conn):
        _put_conn(conn)

    def init_db(self):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id           SERIAL PRIMARY KEY,
                    title        TEXT NOT NULL,
                    body         TEXT,
                    url          TEXT UNIQUE NOT NULL,
                    published_at TIMESTAMP,
                    source       TEXT NOT NULL,
                    category     TEXT,
                    scraped_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    body_fetched INTEGER DEFAULT 0,
                    body_cleaned TEXT,
                    embedding    BYTEA,
                    cluster_id   INTEGER,
                    image_url    TEXT
                )
            ''')
            cur.execute('ALTER TABLE articles ADD COLUMN IF NOT EXISTS image_url TEXT')
            # embedding_model records which model produced the stored vector.
            # Without this, changing EMBEDDING_MODEL would silently mix
            # vectors from two different, non-comparable embedding spaces
            # into the same clustering run with no error — HDBSCAN would
            # treat them as directly comparable when they mathematically
            # aren't. Recording the model lets get_articles_without_embedding
            # detect the mismatch and safely re-embed under the new model.
            # embedding_source_hash records a hash of the exact text that
            # was embedded, so a body update (re-cleaning, a source
            # correction) can also be detected as making the stored vector
            # stale, independent of any model change.
            # Human override for a clustering mistake. Automated removal of
            # wrongly-grouped articles was built, measured and rejected: the
            # wrong article routinely scores HIGHER than a correct one (a
            # Muay Thai report sat at 0.963 inside a javelin cluster whose
            # genuine members scored 0.925), and an LLM judge over the
            # undecidable band made the multi-outlet cluster count worse than
            # doing nothing. What remains is targeted human correction — rare,
            # deliberate, and reversible.
            cur.execute('ALTER TABLE articles ADD COLUMN IF NOT EXISTS '
                        'cluster_excluded BOOLEAN DEFAULT FALSE')
            cur.execute('ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding_model TEXT')
            cur.execute('ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding_source_hash TEXT')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS custom_sources (
                    id           SERIAL PRIMARY KEY,
                    name         TEXT NOT NULL,
                    url          TEXT UNIQUE NOT NULL,
                    added_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_scraped TIMESTAMP,
                    active       INTEGER DEFAULT 1
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS cluster_labels (
                    cluster_id  INTEGER PRIMARY KEY,
                    event_title TEXT NOT NULL,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Topics: the coarser grouping layer *above* event clusters —
            # several related events ("IMF loan tranche approved", "IMF
            # conditions debated in parliament") rolled up under one theme.
            # Membership lives on cluster_labels.topic_id rather than on
            # articles, so an article's topic is reached via its cluster and
            # no article row is ever rewritten by topic clustering. See
            # clusterer.py's ClusterTopicBuilder.
            cur.execute('''
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id    SERIAL PRIMARY KEY,
                    topic_title TEXT NOT NULL,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cur.execute('ALTER TABLE cluster_labels ADD COLUMN IF NOT EXISTS topic_id INTEGER')
            # Fingerprint of a topic's member cluster ids. Relabelling is the
            # only expensive part of the topic pass (it calls an LLM), so it
            # runs only when this changes rather than every cycle.
            cur.execute('ALTER TABLE topics ADD COLUMN IF NOT EXISTS member_hash TEXT')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_cluster_labels_topic ON cluster_labels(topic_id)')
            # Remembered LLM answers for topic joins. The model is not
            # deterministic, so re-asking a settled question every run made
            # topic membership flap; this makes an answer stick until the
            # topic itself changes. See get_join_verdicts.
            cur.execute('''
                CREATE TABLE IF NOT EXISTS cluster_member_checks (
                    article_id  INTEGER NOT NULL,
                    peers_hash  TEXT    NOT NULL,
                    verdict     BOOLEAN NOT NULL,
                    checked_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (article_id, peers_hash)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS topic_join_checks (
                    cluster_id  INTEGER NOT NULL,
                    topic_id    INTEGER NOT NULL,
                    member_hash TEXT    NOT NULL,
                    verdict     BOOLEAN NOT NULL,
                    checked_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (cluster_id, topic_id, member_hash)
                )
            ''')

            # ── Claim retrieval (/search) ──────────────────────────────────
            # Additive only: `embedding` BYTEA stays the source of truth that
            # clusterer.py reads, and `embedding_vec` mirrors it purely so
            # similarity search can run *inside* Postgres. That keeps the
            # 2,878-vector scan off the wire (~4MB per query if fetched into
            # Python) and means the search path never needs numpy.
            #
            # `fts` is a GENERATED column: Postgres recomputes it on every
            # title/body_cleaned write by itself, so no pipeline stage has to
            # remember to maintain it.
            self._search_available = True
            try:
                from config import EMBEDDING_DIMENSION
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

                # The column is fixed-width, so switching embedding provider
                # (384 local vs 768 Gemini) means recreating it. Dropping is
                # safe: vector spaces are not comparable across models, so
                # the old values are worthless the moment the provider
                # changes, and the embedding_model check re-embeds every
                # article on the next run anyway.
                cur.execute("""
                    SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = 'articles'::regclass AND attname = 'embedding_vec'
                """)
                row = cur.fetchone()
                existing_dim = (row[0] if row else None)
                if existing_dim is not None and existing_dim != EMBEDDING_DIMENSION:
                    logger.warning(
                        f"embedding_vec is vector({existing_dim}) but the configured "
                        f"provider needs vector({EMBEDDING_DIMENSION}) — recreating the "
                        f"column; articles will be re-embedded automatically."
                    )
                    cur.execute('DROP INDEX IF EXISTS idx_articles_vec')
                    cur.execute('ALTER TABLE articles DROP COLUMN IF EXISTS embedding_vec')

                cur.execute(
                    f'ALTER TABLE articles ADD COLUMN IF NOT EXISTS '
                    f'embedding_vec vector({EMBEDDING_DIMENSION})'
                )
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_articles_vec ON articles
                    USING hnsw (embedding_vec vector_cosine_ops)
                ''')
            except Exception as e:
                # A managed Postgres without pgvector can still run everything
                # else; only /search degrades, and api.py reports that rather
                # than 500ing. Rolled back so the rest of init_db can proceed
                # on a clean transaction.
                conn.rollback()
                self._search_available = False
                logger.warning(f"pgvector unavailable — /search dense channel disabled: {e}")

            cur.execute('''
                ALTER TABLE articles ADD COLUMN IF NOT EXISTS fts tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('english',
                        coalesce(title, '') || ' ' || coalesce(body_cleaned, ''))
                ) STORED
            ''')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN (fts)')

            # Queries we had no answer for. Two uses: a coverage-gap signal
            # (what are people asking about that our sources don't carry?) and
            # a growing set of real retrieval failures to analyse.
            cur.execute('''
                CREATE TABLE IF NOT EXISTS unmatched_queries (
                    id          SERIAL PRIMARY KEY,
                    query_text  TEXT NOT NULL,
                    best_score  REAL,
                    asked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            for idx in [
                'CREATE INDEX IF NOT EXISTS idx_url          ON articles(url)',
                'CREATE INDEX IF NOT EXISTS idx_source       ON articles(source)',
                'CREATE INDEX IF NOT EXISTS idx_category     ON articles(category)',
                'CREATE INDEX IF NOT EXISTS idx_published    ON articles(published_at)',
                'CREATE INDEX IF NOT EXISTS idx_scraped      ON articles(scraped_at)',
                'CREATE INDEX IF NOT EXISTS idx_cluster      ON articles(cluster_id)',
                'CREATE INDEX IF NOT EXISTS idx_body_fetched ON articles(body_fetched)',
            ]:
                cur.execute(idx)
        conn.commit()
        self._release(conn)

    # ── write ──────────────────────────────────────────────────────────────

    def save_article(self, article: Dict) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO articles (title, body, url, published_at, source, category, image_url, body_fetched)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                ''', (
                    article['title'][:500],
                    article.get('body', '')[:50000],
                    article['url'],
                    article.get('published_at'),
                    article['source'],
                    article.get('category', 'general'),
                    article.get('image_url') or None,
                    # sources that supply the body up front (e.g. JSON APIs whose
                    # article pages render client-side) mark it fetched so the
                    # extractor step doesn't overwrite it with an empty result
                    int(article.get('body_fetched', 0)),
                ))
                saved = cur.rowcount > 0
            conn.commit()
            self._release(conn)
            return saved
        except Exception as e:
            logger.error(f"Error saving article: {e}")
            return False

    def save_batch(self, articles: List[Dict]) -> int:
        return sum(1 for a in articles if self.save_article(a))

    def update_body(self, article_id: int, body: str, image_url: str = None):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET body=%s, body_fetched=1, image_url=COALESCE(%s, image_url) WHERE id=%s",
                (body[:50000], image_url or None, article_id)
            )
        conn.commit()
        self._release(conn)

    def get_articles_to_clean(self) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, body FROM articles WHERE body_fetched=1 AND body_cleaned IS NULL AND body != ''"
            )
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def update_article_title(self, article_id: int, title: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("UPDATE articles SET title=%s WHERE id=%s", (title[:500], article_id))
        conn.commit()
        self._release(conn)

    def update_article_category(self, article_id: int, category: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("UPDATE articles SET category=%s WHERE id=%s", (category, article_id))
        conn.commit()
        self._release(conn)

    def update_cleaned_body(self, article_id: int, cleaned: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET body_cleaned=%s WHERE id=%s",
                (cleaned, article_id)
            )
        conn.commit()
        self._release(conn)

    def update_embedding(self, article_id: int, embedding_bytes: bytes,
                          embedding_model: str, source_hash: str,
                          embedding_list: Optional[List[float]] = None):
        """
        Stores the vector twice, deliberately: `embedding` (BYTEA) is what
        clusterer.py reads and stays the source of truth, while
        `embedding_vec` is the pgvector mirror that /search queries. Passing
        embedding_list keeps them written in the same transaction so the two
        can never disagree about a given article.
        """
        conn = self._connect()
        with conn.cursor() as cur:
            if embedding_list is not None:
                cur.execute(
                    "UPDATE articles SET embedding=%s, embedding_model=%s, "
                    "embedding_source_hash=%s, embedding_vec=%s WHERE id=%s",
                    (psycopg2.Binary(embedding_bytes), embedding_model, source_hash,
                     str(embedding_list), article_id)
                )
            else:
                cur.execute(
                    "UPDATE articles SET embedding=%s, embedding_model=%s, "
                    "embedding_source_hash=%s WHERE id=%s",
                    (psycopg2.Binary(embedding_bytes), embedding_model, source_hash, article_id)
                )
        conn.commit()
        self._release(conn)

    def backfill_embedding_vec(self, batch_size: int = 500) -> int:
        """
        One-off migration: populate embedding_vec from the existing BYTEA
        column for articles embedded before /search existed. Converts in
        Postgres-side batches rather than one big UPDATE so a large corpus
        doesn't hold a single long transaction open against Neon.
        Idempotent — only touches rows where embedding_vec IS NULL.
        """
        import numpy as np
        total = 0
        while True:
            conn = self._connect()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, embedding FROM articles "
                    "WHERE embedding IS NOT NULL AND embedding_vec IS NULL LIMIT %s",
                    (batch_size,)
                )
                rows = cur.fetchall()
                if not rows:
                    self._release(conn)
                    break
                payload = [
                    (str(np.frombuffer(bytes(r['embedding']), dtype='float32').tolist()), r['id'])
                    for r in rows
                ]
                psycopg2.extras.execute_batch(
                    cur, "UPDATE articles SET embedding_vec=%s WHERE id=%s", payload
                )
            conn.commit()
            self._release(conn)
            total += len(rows)
            logger.info(f"  backfilled embedding_vec for {total} articles...")
        return total

    def update_cluster_ids(self, assignments: List[Tuple[int, int]]):
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE articles SET cluster_id=%s WHERE id=%s",
                assignments
            )
        conn.commit()
        self._release(conn)

    def save_cluster_label(self, cluster_id: int, title: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO cluster_labels (cluster_id, event_title, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (cluster_id) DO UPDATE
                SET event_title=EXCLUDED.event_title, updated_at=CURRENT_TIMESTAMP
            ''', (cluster_id, title))
        conn.commit()
        self._release(conn)

    def save_cluster_labels_bulk(self, labels: List[Tuple[int, str]]):
        """save_cluster_label for many clusters in one round trip.

        Same reason as get_cluster_members_bulk: once single-outlet stories
        became their own clusters, a run labels ~850 clusters instead of
        ~320, and against a hosted database the per-statement latency
        dominates.
        """
        if not labels:
            return
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                'INSERT INTO cluster_labels (cluster_id, event_title, updated_at) '
                'VALUES %s '
                'ON CONFLICT (cluster_id) DO UPDATE '
                'SET event_title=EXCLUDED.event_title, updated_at=CURRENT_TIMESTAMP',
                [(cid, title, ) for cid, title in labels],
                template='(%s, %s, CURRENT_TIMESTAMP)',
                page_size=500,
            )
        conn.commit()
        self._release(conn)

    def get_cluster_labels(self) -> dict:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT cluster_id, event_title FROM cluster_labels')
            rows = cur.fetchall()
        self._release(conn)
        return {r['cluster_id']: r['event_title'] for r in rows}

    # ── topics (the grouping layer above event clusters) ───────────────────

    def get_topic_assignments(self) -> Dict[int, int]:
        """{cluster_id: topic_id} for every cluster currently in a topic —
        the previous run's assignment, which topic-identity matching needs to
        recover so an unchanged theme keeps its topic_id (same reasoning as
        get_recent_embeddings recovering cluster_id for events)."""
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT cluster_id, topic_id FROM cluster_labels WHERE topic_id IS NOT NULL')
            rows = cur.fetchall()
        self._release(conn)
        return {r['cluster_id']: r['topic_id'] for r in rows}

    def create_topic(self, title: str, member_hash: str = None) -> int:
        """Mint a new topic id. SERIAL rather than MAX(topic_id)+1 so two
        concurrent callers can't be handed the same id."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO topics (topic_title, member_hash) VALUES (%s, %s) RETURNING topic_id',
                (title, member_hash)
            )
            (topic_id,) = cur.fetchone()
        conn.commit()
        self._release(conn)
        return topic_id

    def update_topic_title(self, topic_id: int, title: str, member_hash: str = None):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE topics SET topic_title=%s, member_hash=COALESCE(%s, member_hash), '
                'updated_at=CURRENT_TIMESTAMP WHERE topic_id=%s',
                (title, member_hash, topic_id)
            )
        conn.commit()
        self._release(conn)

    def get_member_verdicts(self) -> Dict[Tuple[int, str], bool]:
        """Remembered answers to "does article X belong in this cluster?",
        keyed by (article_id, fingerprint of the siblings it was shown with).

        Same reasoning as get_join_verdicts: the model does not answer
        identically twice, and cluster membership flapping every run would
        move article and outlet counts with no news behind it.
        """
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT article_id, peers_hash, verdict FROM cluster_member_checks')
            rows = cur.fetchall()
        self._release(conn)
        return {(r[0], r[1]): r[2] for r in rows}

    def save_member_verdicts(self, verdicts: List[Tuple[int, str, bool]]):
        if not verdicts:
            return
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                'INSERT INTO cluster_member_checks '
                '(article_id, peers_hash, verdict, checked_at) VALUES %s '
                'ON CONFLICT (article_id, peers_hash) DO UPDATE '
                'SET verdict=EXCLUDED.verdict, checked_at=CURRENT_TIMESTAMP',
                verdicts,
                template='(%s, %s, %s, CURRENT_TIMESTAMP)',
                page_size=500,
            )
        conn.commit()
        self._release(conn)

    def get_join_verdicts(self) -> Dict[Tuple[int, int, str], bool]:
        """
        Remembered answers to "does cluster X belong in topic Y, as topic Y
        currently stands?", keyed by (cluster_id, topic_id, member_hash).

        Without this the join stage re-asked the same ~380 questions every
        run, and the model does not answer identically every time — observed
        across consecutive hourly runs with no new articles at all:
        150, 142, 117, 109, 106, 102, 96, 90 accepted out of the same ~381.
        Topic membership was churning purely from LLM non-determinism.

        member_hash is part of the key so a topic that GAINS or LOSES a core
        member is re-asked; a topic that has not changed is not.
        """
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT cluster_id, topic_id, member_hash, verdict '
                        'FROM topic_join_checks')
            rows = cur.fetchall()
        self._release(conn)
        return {(r[0], r[1], r[2]): r[3] for r in rows}

    def save_join_verdicts(self, verdicts: List[Tuple[int, int, str, bool]]):
        if not verdicts:
            return
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                'INSERT INTO topic_join_checks '
                '(cluster_id, topic_id, member_hash, verdict, checked_at) VALUES %s '
                'ON CONFLICT (cluster_id, topic_id, member_hash) DO UPDATE '
                'SET verdict=EXCLUDED.verdict, checked_at=CURRENT_TIMESTAMP',
                verdicts,
                template='(%s, %s, %s, %s, CURRENT_TIMESTAMP)',
                page_size=500,
            )
        conn.commit()
        self._release(conn)

    def prune_join_verdicts(self, keep_days: int = 30) -> int:
        """Drop answers about topics that no longer exist, and stale rows.
        Bounded maintenance so this table cannot grow without limit."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM topic_join_checks
                WHERE topic_id NOT IN (SELECT topic_id FROM topics)
                   OR checked_at < CURRENT_TIMESTAMP - make_interval(days => %s)
            """, (keep_days,))
            n = cur.rowcount
        conn.commit()
        self._release(conn)
        return n

    def get_topic_member_hashes(self) -> Dict[int, str]:
        """{topic_id: member_hash} — lets the topic pass skip relabelling
        topics whose membership is unchanged since last run."""
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT topic_id, member_hash FROM topics WHERE member_hash IS NOT NULL')
            rows = cur.fetchall()
        self._release(conn)
        return {r['topic_id']: r['member_hash'] for r in rows}

    def set_cluster_topics(self, assignments: List[Tuple[Optional[int], int]]):
        """Bulk (topic_id, cluster_id) writes. topic_id=None clears a
        cluster's topic — what a cluster that no longer groups with anything
        gets, rather than being left pointing at a topic it has left."""
        if not assignments:
            return
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                'UPDATE cluster_labels SET topic_id=%s WHERE cluster_id=%s',
                assignments
            )
        conn.commit()
        self._release(conn)

    def delete_empty_topics(self) -> int:
        """Drop topic rows no cluster points at any more — a topic whose
        member events all moved elsewhere is not a topic."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('''
                DELETE FROM topics t
                WHERE NOT EXISTS (
                    SELECT 1 FROM cluster_labels cl WHERE cl.topic_id = t.topic_id
                )
            ''')
            deleted = cur.rowcount
        conn.commit()
        self._release(conn)
        return deleted

    def get_topics(self, min_clusters: int = 2) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('''
                SELECT
                    t.topic_id,
                    t.topic_title,
                    COUNT(DISTINCT cl.cluster_id) as cluster_count,
                    COUNT(DISTINCT (a.source, LOWER(TRIM(a.title)))) as article_count,
                    STRING_AGG(DISTINCT a.source, ',') as sources,
                    MAX(a.published_at) as latest_at,
                    (SELECT a3.image_url FROM articles a3
                     JOIN cluster_labels cl3 ON cl3.cluster_id = a3.cluster_id
                     WHERE cl3.topic_id = t.topic_id
                       AND a3.image_url IS NOT NULL AND a3.image_url != ''
                     ORDER BY a3.published_at DESC NULLS LAST LIMIT 1) as topic_image,
                    (SELECT a4.category FROM articles a4
                     JOIN cluster_labels cl4 ON cl4.cluster_id = a4.cluster_id
                     WHERE cl4.topic_id = t.topic_id AND a4.category IS NOT NULL
                     GROUP BY a4.category ORDER BY COUNT(*) DESC LIMIT 1) as category
                FROM topics t
                JOIN cluster_labels cl ON cl.topic_id = t.topic_id
                JOIN articles a ON a.cluster_id = cl.cluster_id
                WHERE a.body IS NOT NULL AND a.body != ''
                GROUP BY t.topic_id, t.topic_title
                HAVING COUNT(DISTINCT cl.cluster_id) >= %s
                ORDER BY latest_at DESC NULLS LAST
            ''', (min_clusters,))
            rows = cur.fetchall()
        self._release(conn)
        result = []
        for r in rows:
            d = dict(r)
            d['sources'] = d['sources'].split(',') if d['sources'] else []
            if d.get('latest_at') and hasattr(d['latest_at'], 'isoformat'):
                d['latest_at'] = d['latest_at'].isoformat()
            result.append(d)
        return result

    # ── claim retrieval (/search) ──────────────────────────────────────────

    def search_dense(self, query_vec: List[float], limit: int = 50) -> List[Dict]:
        """
        Nearest articles by cosine distance, computed in Postgres via
        pgvector's `<=>` operator. Only clustered articles with a real body
        are eligible — an unclustered or body-less article can't be reported
        as an event match. Returns rows ordered best-first with `score` as
        cosine *similarity* (1 - distance), matching the convention used
        everywhere else in this codebase.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('''
                SELECT id, cluster_id, title, source, url, published_at,
                       1 - (embedding_vec <=> %s::vector) AS score
                FROM articles
                WHERE embedding_vec IS NOT NULL
                  AND cluster_id IS NOT NULL AND cluster_id >= 0
                  AND body IS NOT NULL AND body != ''
                ORDER BY embedding_vec <=> %s::vector
                LIMIT %s
            ''', (str(query_vec), str(query_vec), limit))
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def search_lexical(self, query_text: str, limit: int = 50) -> List[Dict]:
        """
        Keyword channel over the generated `fts` column. Uses ts_rank_cd
        (cover density) rather than plain ts_rank so proximity of the query
        terms counts, which matters for multi-word claims. plainto_tsquery
        rather than to_tsquery so arbitrary user text can't raise a syntax
        error on stray punctuation.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('''
                SELECT id, cluster_id, title, source, url, published_at,
                       ts_rank_cd(fts, plainto_tsquery('english', %s)) AS score
                FROM articles
                WHERE fts @@ plainto_tsquery('english', %s)
                  AND cluster_id IS NOT NULL AND cluster_id >= 0
                  AND body IS NOT NULL AND body != ''
                ORDER BY score DESC
                LIMIT %s
            ''', (query_text, query_text, limit))
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def get_cluster_summaries(self, cluster_ids: List[int]) -> Dict[int, Dict]:
        """
        Display metadata for a specific set of clusters — label, topic,
        distinct-article count, sources. Same shape the /clusters endpoints
        return, so a search result is interchangeable with a feed item.
        """
        if not cluster_ids:
            return {}
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('''
                SELECT a.cluster_id,
                       COUNT(DISTINCT (a.source, LOWER(TRIM(a.title)))) AS article_count,
                       STRING_AGG(DISTINCT a.source, ',') AS sources,
                       MAX(a.published_at) AS latest_at,
                       (SELECT a2.category FROM articles a2
                        WHERE a2.cluster_id = a.cluster_id AND a2.category IS NOT NULL
                        GROUP BY a2.category ORDER BY COUNT(*) DESC LIMIT 1) AS category,
                       cl.event_title, cl.topic_id, t.topic_title
                FROM articles a
                LEFT JOIN cluster_labels cl ON cl.cluster_id = a.cluster_id
                LEFT JOIN topics t ON t.topic_id = cl.topic_id
                WHERE a.cluster_id = ANY(%s)
                  AND a.body IS NOT NULL AND a.body != ''
                GROUP BY a.cluster_id, cl.event_title, cl.topic_id, t.topic_title
            ''', (cluster_ids,))
            rows = cur.fetchall()
        self._release(conn)
        out = {}
        for r in rows:
            d = dict(r)
            d['sources'] = d['sources'].split(',') if d['sources'] else []
            if d.get('latest_at') and hasattr(d['latest_at'], 'isoformat'):
                d['latest_at'] = d['latest_at'].isoformat()
            out[d['cluster_id']] = d
        return out

    def log_unmatched_query(self, query_text: str, best_score: float):
        """Best-effort — a logging failure must never fail the user's search."""
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO unmatched_queries (query_text, best_score) VALUES (%s, %s)",
                    (query_text[:2000], best_score)
                )
            conn.commit()
            self._release(conn)
        except Exception as e:
            logger.warning(f"Could not log unmatched query: {e}")

    def get_topic_for_cluster(self, cluster_id: int) -> Optional[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('''
                SELECT t.topic_id, t.topic_title
                FROM cluster_labels cl
                JOIN topics t ON t.topic_id = cl.topic_id
                WHERE cl.cluster_id = %s
            ''', (cluster_id,))
            row = cur.fetchone()
        self._release(conn)
        return dict(row) if row else None

    def get_topic(self, topic_id: int) -> Optional[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT topic_id, topic_title FROM topics WHERE topic_id=%s', (topic_id,))
            row = cur.fetchone()
        self._release(conn)
        return dict(row) if row else None

    def get_article_title(self, article_id: int) -> str:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT title FROM articles WHERE id=%s', (article_id,))
            row = cur.fetchone()
        self._release(conn)
        return row['title'] if row else ''

    def get_article_titles_batch(self, article_ids: List[int]) -> Dict[int, str]:
        if not article_ids:
            return {}
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute('SELECT id, title FROM articles WHERE id = ANY(%s)', (article_ids,))
            rows = cur.fetchall()
        self._release(conn)
        return {r['id']: r['title'] for r in rows}

    # ── read ───────────────────────────────────────────────────────────────

    def url_exists(self, url: str) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute('SELECT id FROM articles WHERE url=%s', (url,))
                row = cur.fetchone()
            self._release(conn)
            return row is not None
        except Exception as e:
            logger.error(f"Error checking URL: {e}")
            return False

    def get_articles_without_image(self, limit: int = 200) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, url FROM articles WHERE body_fetched=1 AND (image_url IS NULL OR image_url='') LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def update_image(self, article_id: int, image_url: str):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET image_url=%s WHERE id=%s",
                (image_url or None, article_id)
            )
        conn.commit()
        self._release(conn)

    def get_articles_without_body(self, limit: int = 50) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, url, title, source FROM articles WHERE body_fetched=0 LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def get_articles_without_embedding(self, current_model: str, limit: int = 500) -> List[Dict]:
        """
        Articles with no embedding yet, *or* whose stored embedding_model
        doesn't match current_model.

        Body-less articles are excluded. `body_fetched=1` only means
        extraction was *attempted* — when it failed, the article would be
        embedded on its title alone, and a title-only vector is weak enough
        that it pollutes clustering rather than helping it. Such articles
        stay in the table (the URL record still prevents re-scraping) but
        never reach a cluster.

        The rest of the docstring — the latter is what makes changing
        EMBEDDING_MODEL in config.py safe: every article is automatically
        picked up for re-embedding under the new model on the next run,
        instead of silently keeping a stale vector from a different,
        non-comparable embedding space forever.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, category, COALESCE(body_cleaned, '') as body_cleaned "
                "FROM articles "
                "WHERE body_fetched=1 AND body IS NOT NULL AND body <> '' "
                "AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %s) "
                "LIMIT %s",
                (current_model, limit)
            )
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> List[Tuple[int, bytes]]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, embedding FROM articles WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
        self._release(conn)
        return [(r['id'], bytes(r['embedding'])) for r in rows]

    def get_recent_embeddings(self, days: int, embedding_model: str = None) -> List[Dict]:
        """
        Embedded articles from the last `days`, with whatever cluster_id they
        already carry (NULL/-1 for none). Used by the clusterer to re-fit only
        a recent window instead of the whole historical corpus each run, and
        to recover each article's *previous* assignment so new cluster
        identities can be matched back to it (see clusterer.py). `source` and
        `effective_date` are included for the cluster-quality refinement pass
        (outlet-dominance and time-span checks) that runs before that match.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, embedding, cluster_id, source,
                       COALESCE(published_at, scraped_at) AS effective_date
                FROM articles
                WHERE embedding IS NOT NULL
                  AND body IS NOT NULL AND body <> ''
                  AND (%s IS NULL OR embedding_model = %s)
                  AND COALESCE(published_at, scraped_at) >= NOW() - (%s || ' days')::interval
            """, (embedding_model, embedding_model, days))
            rows = cur.fetchall()
        self._release(conn)
        return [
            {
                'id': r['id'], 'embedding': bytes(r['embedding']),
                'cluster_id': r['cluster_id'], 'source': r['source'],
                'effective_date': r['effective_date'],
            }
            for r in rows
        ]

    def set_cluster_excluded(self, article_id: int, excluded: bool = True) -> bool:
        """Mark an article as not belonging to whatever cluster it lands in.
        It keeps its row and its url record; clustering simply refuses to
        group it, so it becomes a single-article cluster of its own."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('UPDATE articles SET cluster_excluded=%s WHERE id=%s',
                        (excluded, article_id))
            ok = cur.rowcount > 0
        conn.commit()
        self._release(conn)
        return ok

    def get_excluded_article_ids(self) -> set:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT id FROM articles WHERE cluster_excluded IS TRUE')
            ids = {r[0] for r in cur.fetchall()}
        self._release(conn)
        return ids

    def get_max_article_id(self) -> int:
        """Highest article id in the table — the watermark a consumer passes
        back as `since_id` to fetch only what has changed since."""
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT COALESCE(MAX(id), 0) FROM articles')
            n = cur.fetchone()[0]
        self._release(conn)
        return int(n)

    def get_max_cluster_id(self) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM articles")
            (max_id,) = cur.fetchone()
        self._release(conn)
        return max_id

    def get_recent_embedding_meta(self, days: int, embedding_model: str = None) -> List[Dict]:
        """Everything get_recent_embeddings returns EXCEPT the vectors.

        Splitting the vectors out is what makes an embedding cache possible.
        Metadata for a 14-day window is a couple of hundred kilobytes; the
        vectors are ~8 MB, and re-downloading them every hour was the single
        largest consumer of the database's network allowance (measured: 32 MB
        per run across three queries, 22.7 GB/month against a 5 GB limit).
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, cluster_id, source,
                       COALESCE(published_at, scraped_at) AS effective_date
                FROM articles
                WHERE embedding IS NOT NULL
                  AND body IS NOT NULL AND body <> ''
                  AND (%s IS NULL OR embedding_model = %s)
                  AND COALESCE(published_at, scraped_at) >= NOW() - (%s || ' days')::interval
            """, (embedding_model, embedding_model, days))
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def get_embeddings_by_ids(self, article_ids: List[int]) -> Dict[int, bytes]:
        """Vectors for specific articles — the cache-miss path."""
        if not article_ids:
            return {}
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT id, embedding FROM articles '
                        'WHERE id = ANY(%s) AND embedding IS NOT NULL',
                        (list(article_ids),))
            rows = cur.fetchall()
        self._release(conn)
        return {r[0]: bytes(r[1]) for r in rows}

    def get_cluster_membership(self, embedding_model: str) -> Dict[int, List[int]]:
        """{cluster_id: [article_id]} for every clustered article, no vectors.

        Lets the topic pass rebuild centroids from cached vectors instead of
        pulling 12 MB of embeddings out of the database on every run.
        """
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cluster_id, id FROM articles
                WHERE cluster_id IS NOT NULL AND cluster_id >= 0
                  AND embedding IS NOT NULL
                  AND body IS NOT NULL AND body <> ''
                  AND embedding_model = %s
            """, (embedding_model,))
            rows = cur.fetchall()
        self._release(conn)
        out: Dict[int, List[int]] = {}
        for cid, aid in rows:
            out.setdefault(cid, []).append(aid)
        return out

    def get_all_cluster_embeddings(self, embedding_model: str) -> List[Dict]:
        """
        (cluster_id, embedding) for every clustered article in one query, so
        the topic pass can build a centroid for EVERY cluster rather than
        only the ones inside the current clustering window.

        Why that matters: a topic is a *relationship between events*, and it
        is recomputed under whatever thresholds are configured now. Leaving
        out-of-window clusters untouched froze their topic_id at whatever an
        older threshold decided — 48% of assignments were such fossils, which
        is why changing the linkage appeared to have no effect on them.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT cluster_id, embedding FROM articles
                WHERE cluster_id IS NOT NULL AND cluster_id >= 0
                  AND embedding IS NOT NULL
                  AND body IS NOT NULL AND body <> ''
                  AND embedding_model = %s
            """, (embedding_model,))
            rows = cur.fetchall()
        self._release(conn)
        return [{'cluster_id': r['cluster_id'], 'embedding': bytes(r['embedding'])}
                for r in rows]

    def get_articles_for_clusters(self, cluster_ids: List[int],
                                  per_cluster: int = 10,
                                  full_body: bool = False) -> Dict[int, List[Dict]]:
        """Articles for many clusters in ONE query, newest first, capped per
        cluster.

        Exists so /clusters?include=articles stays two queries no matter how
        many clusters are returned. Looping get_cluster_articles() would be
        one round trip per cluster — the classic N+1 — and against a hosted
        database the latency, not the work, would dominate the response.

        The cap is applied in SQL with a window function rather than by
        fetching everything and slicing in Python, so a cluster with 200
        articles does not pull 200 rows over the wire to show 10.
        """
        if not cluster_ids:
            return {}
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Full text is opt-in: a bias/fact-check consumer needs the real
            # wording, but a feed rendering 20 clusters does not want 200
            # full articles on the wire.
            body_expr = ("COALESCE(a.body_cleaned, a.body, '')" if full_body
                         else "LEFT(COALESCE(a.body_cleaned, a.body, ''), 400)")
            cur.execute(f"""
                SELECT * FROM (
                    SELECT a.cluster_id, a.id, a.title, a.source, a.url,
                           a.published_at, a.image_url,
                           {body_expr} AS body,
                           ROW_NUMBER() OVER (
                               PARTITION BY a.cluster_id
                               ORDER BY a.published_at DESC NULLS LAST, a.id DESC
                           ) AS rn,
                           -- Second numbering, over (outlet, headline), so a
                           -- republished story is shown once. See
                           -- get_cluster_articles for why URL dedup misses it.
                           ROW_NUMBER() OVER (
                               PARTITION BY a.cluster_id, a.source,
                                            LOWER(TRIM(a.title))
                               ORDER BY a.published_at DESC NULLS LAST, a.id DESC
                           ) AS dup_rn
                    FROM articles a
                    WHERE a.cluster_id = ANY(%s)
                      AND a.body IS NOT NULL AND a.body <> ''
                ) t WHERE dup_rn = 1 AND rn <= %s
            """, (list(cluster_ids), per_cluster))
            rows = cur.fetchall()
        self._release(conn)
        out: Dict[int, List[Dict]] = {}
        for r in rows:
            out.setdefault(r['cluster_id'], []).append({
                'id': r['id'], 'title': r['title'], 'source': r['source'],
                'url': r['url'], 'published_at': r['published_at'],
                'image_url': r['image_url'], 'body': r['body'],
            })
        return out

    def get_cluster_member_titles(self, cluster_ids: List[int],
                                  embedding_model: str = None) -> Dict[int, List[Dict]]:
        """get_cluster_members_bulk without the vectors.

        Labelling needs the titles and the ids; the vectors it also used come
        from the in-process cache instead. That removes the third full
        embedding download per run — the largest of the three at ~12 MB.
        """
        if not cluster_ids:
            return {}
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT cluster_id, id, title FROM articles "
                "WHERE cluster_id = ANY(%s) AND embedding IS NOT NULL "
                "AND (%s IS NULL OR embedding_model = %s)",
                (list(cluster_ids), embedding_model, embedding_model)
            )
            rows = cur.fetchall()
        self._release(conn)
        out: Dict[int, List[Dict]] = {}
        for r in rows:
            out.setdefault(r['cluster_id'], []).append(
                {'id': r['id'], 'title': r['title']})
        return out

    def get_cluster_members_bulk(self, cluster_ids: List[int],
                                 embedding_model: str = None) -> Dict[int, List[Dict]]:
        """get_cluster_member_data for many clusters in ONE round trip.

        Labelling used to issue a query per cluster. That was fine at ~320
        clusters a run, but promoting single-outlet stories to their own
        clusters roughly triples the count, and against a hosted Postgres the
        per-query latency — not the query itself — becomes the dominant cost
        of a run.
        """
        if not cluster_ids:
            return {}
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT cluster_id, id, title, embedding FROM articles "
                "WHERE cluster_id = ANY(%s) AND embedding IS NOT NULL "
                "AND (%s IS NULL OR embedding_model = %s)",
                (list(cluster_ids), embedding_model, embedding_model)
            )
            rows = cur.fetchall()
        self._release(conn)
        out: Dict[int, List[Dict]] = {}
        for r in rows:
            out.setdefault(r['cluster_id'], []).append(
                {'id': r['id'], 'title': r['title'], 'embedding': bytes(r['embedding'])})
        return out

    def get_cluster_member_data(self, cluster_id: int,
                                 embedding_model: str = None) -> List[Dict]:
        """Full membership (id, title, embedding) of a persistent cluster —
        used to (re)generate its label from its whole history, not just
        whichever slice of it fell inside this run's re-fit window.

        Same single-model filter as get_recent_embeddings, and for the same
        reason: a cluster's history can straddle a provider change, and these
        vectors get stacked."""
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, embedding FROM articles "
                "WHERE cluster_id=%s AND embedding IS NOT NULL "
                "AND (%s IS NULL OR embedding_model = %s)",
                (cluster_id, embedding_model, embedding_model)
            )
            rows = cur.fetchall()
        self._release(conn)
        return [{'id': r['id'], 'title': r['title'], 'embedding': bytes(r['embedding'])}
                for r in rows]

    # Allowlist, not string interpolation of caller input: `sort` reaches the
    # ORDER BY clause, which can't be parameterised, so the *only* safe shape
    # is mapping a known key to a fixed fragment written here.
    _CLUSTER_SORTS = {
        # Newest coverage first — the default news-feed reading order.
        'latest': 'latest_at DESC NULLS LAST',
        # Most independently-corroborated events first: how many distinct
        # outlets carried the story, which is the better "biggest story"
        # signal than raw article count (one outlet publishing five updates
        # is not five outlets agreeing).
        'sources': 'source_count DESC, article_count DESC, latest_at DESC',
        # Most total coverage first, counting deduplicated articles.
        'articles': 'article_count DESC, source_count DESC, latest_at DESC',
    }

    def get_clusters(self, min_size: int = 2, category: str = None,
                     topic_id: int = None, sort: str = 'latest',
                     min_sources: int = 1, since_id: int = 0,
                     limit: int = None, offset: int = 0) -> List[Dict]:
        """min_size counts distinct ARTICLES, min_sources counts distinct
        OUTLETS. They are not interchangeable: one outlet publishing three
        related pieces satisfies min_size=2 while showing a single outlet
        badge, which reads on the feed as a lone story rather than an event
        several papers covered."""
        conn = self._connect()
        order_by = self._CLUSTER_SORTS.get(sort, self._CLUSTER_SORTS['latest'])
        # Category is filtered on the CLUSTER's majority category, in an outer
        # query — NOT on individual articles in the WHERE clause.
        #
        # Filtering articles was wrong three ways at once. A cluster with one
        # economics article and four politics ones was returned for
        # category=economics, then displayed a POLITICS badge (the badge has
        # always shown the majority), and its article_count and sources were
        # computed from the filtered rows, so a 5-outlet event reported 1.
        # The feed showed politics stories under the Economics tab.
        cat_filter = ""
        topic_filter = (
            "AND a.cluster_id IN (SELECT cluster_id FROM cluster_labels WHERE topic_id=%s)"
            if topic_id is not None else ""
        )
        # Order matters: the filters sit in WHERE, ahead of HAVING's %s.
        params = []
        if topic_id is not None:
            params.append(topic_id)
        params.append(min_size)
        params.append(min_sources)
        # Incremental sync, keyed on ARTICLE id rather than cluster id.
        # Cluster ids persist deliberately — a cluster keeps its id when a
        # new outlet joins it — so paging by cluster id would silently miss
        # exactly the updates a consumer most wants: another paper covering
        # a story it already has. Article ids are SERIAL and append-only, so
        # "this cluster has an article newer than X" catches both brand new
        # events and new coverage of old ones.
        params.append(since_id)
        # Applied to the GROUPED rows, so it matches the badge the UI shows
        # and leaves the per-cluster aggregates counting every article.
        params.append(category)
        params.append(category)
        # Paging in SQL, not in Python. The caller previously received every
        # matching cluster and sliced it — 101 rows built with per-cluster
        # subqueries to display 10.
        page_clause = ''
        if limit is not None:
            page_clause = 'LIMIT %s OFFSET %s'
            params.append(limit)
            params.append(offset)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f'''
                SELECT * FROM (
                SELECT
                    a.cluster_id,
                    COUNT(DISTINCT (a.source, LOWER(TRIM(a.title)))) as article_count,
                    COUNT(DISTINCT a.source) as source_count,
                    STRING_AGG(DISTINCT a.source, ',') as sources,
                    MAX(a.published_at) as latest_at,
                    MAX(a.title) FILTER (
                        WHERE LENGTH(a.title) = (
                            SELECT MAX(LENGTH(a2.title))
                            FROM articles a2
                            WHERE a2.cluster_id = a.cluster_id
                        )
                    ) as fallback_title,
                    (SELECT a3.image_url FROM articles a3
                     WHERE a3.cluster_id = a.cluster_id
                       AND a3.image_url IS NOT NULL AND a3.image_url != ''
                     ORDER BY a3.published_at DESC NULLS LAST LIMIT 1) as cluster_image,
                    (SELECT a4.category FROM articles a4
                     WHERE a4.cluster_id = a.cluster_id
                       AND a4.category IS NOT NULL
                     GROUP BY a4.category ORDER BY COUNT(*) DESC LIMIT 1) as category
                FROM articles a
                WHERE a.cluster_id IS NOT NULL AND a.cluster_id >= 0
                  AND a.body IS NOT NULL AND a.body != ''
                {cat_filter}
                {topic_filter}
                GROUP BY a.cluster_id
                HAVING COUNT(DISTINCT (a.source, LOWER(TRIM(a.title)))) >= %s
                   AND COUNT(DISTINCT a.source) >= %s
                   AND MAX(a.id) > %s
            ) grouped
            WHERE (%s IS NULL OR grouped.category = %s)
            ORDER BY {order_by}
            {page_clause}
            ''', params)
            rows = cur.fetchall()

            cur.execute('''
                SELECT cl.cluster_id, cl.event_title, cl.topic_id, t.topic_title
                FROM cluster_labels cl
                LEFT JOIN topics t ON t.topic_id = cl.topic_id
            ''')
            stored = {
                r['cluster_id']: (r['event_title'], r['topic_id'], r['topic_title'])
                for r in cur.fetchall()
            }

        self._release(conn)
        result = []
        for r in rows:
            d = dict(r)
            d['sources'] = d['sources'].split(',') if d['sources'] else []
            cid = d['cluster_id']
            label, tid, ttitle = stored.get(cid, (None, None, None))
            d['event_title'] = label or d.pop('fallback_title') or ''
            d.pop('fallback_title', None)
            d['topic_id'] = tid
            d['topic_title'] = ttitle
            if d.get('latest_at') and hasattr(d['latest_at'], 'isoformat'):
                d['latest_at'] = d['latest_at'].isoformat()
            result.append(d)
        return result

    def get_cluster_articles(self, cluster_id: int) -> List[Dict]:
        """
        One row per (outlet, headline), newest kept.

        Outlets republish: Ada Derana carried "Advisory issued for strong
        winds and rough seas" on two consecutive days under different URLs,
        so URL-level dedup — which is airtight — cannot see it. Without this
        the same headline appeared twice in a cluster while article_count,
        which has always counted DISTINCT (source, title), said otherwise.

        body_cleaned first: `body` is the raw extraction and still carries
        markup from some sources, which rendered as a literal "<p>" in the
        UI.
        """
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, body, source, published_at, url, cluster_id,
                       category, image_url
                FROM (
                    SELECT id, title,
                           COALESCE(body_cleaned, body) AS body,
                           source, published_at, url, cluster_id, category,
                           image_url,
                           ROW_NUMBER() OVER (
                               PARTITION BY source, LOWER(TRIM(title))
                               ORDER BY published_at DESC NULLS LAST, id DESC
                           ) AS rn
                    FROM articles WHERE cluster_id=%s
                ) t WHERE rn = 1
                ORDER BY published_at DESC NULLS LAST
            """, (cluster_id,))
            rows = cur.fetchall()
        self._release(conn)
        result = []
        for r in rows:
            d = dict(r)
            if d.get('published_at') and hasattr(d['published_at'], 'isoformat'):
                d['published_at'] = d['published_at'].isoformat()
            result.append(d)
        return result

    def get_article_by_id(self, article_id: int) -> Optional[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
            row = cur.fetchone()
        self._release(conn)
        if not row:
            return None
        d = dict(row)
        for field in ('published_at', 'scraped_at'):
            if d.get(field) and hasattr(d[field], 'isoformat'):
                d[field] = d[field].isoformat()
        return d

    def get_articles(self, source: str = None, category: str = None,
                     limit: int = 20, offset: int = 0) -> List[Dict]:
        filters, params = [], []
        if source:
            filters.append("source=%s")
            params.append(source)
        if category:
            filters.append("category=%s")
            params.append(category)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        params += [limit, offset]
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"SELECT id, title, body, source, published_at, url, cluster_id, category, scraped_at "
                f"FROM articles {where} ORDER BY scraped_at DESC LIMIT %s OFFSET %s",
                params
            )
            rows = cur.fetchall()
        self._release(conn)
        result = []
        for r in rows:
            d = dict(r)
            for field in ('published_at', 'scraped_at'):
                if d.get(field) and hasattr(d[field], 'isoformat'):
                    d[field] = d[field].isoformat()
            result.append(d)
        return result

    def get_stats(self) -> Tuple[int, Dict, Dict]:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM articles')
                total = cur.fetchone()[0]
                cur.execute('SELECT source, COUNT(*) FROM articles GROUP BY source')
                by_source = dict(cur.fetchall())
                cur.execute('SELECT category, COUNT(*) FROM articles GROUP BY category')
                by_category = dict(cur.fetchall())
            self._release(conn)
            return total, by_source, by_category
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return 0, {}, {}

    def get_cluster_stats(self) -> Dict:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM articles')
            total = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM articles WHERE cluster_id IS NOT NULL AND cluster_id >= 0')
            clustered = cur.fetchone()[0]
            cur.execute('SELECT COUNT(DISTINCT cluster_id) FROM articles WHERE cluster_id IS NOT NULL AND cluster_id >= 0')
            num_clusters = cur.fetchone()[0]
        self._release(conn)
        return {
            'total_articles': total,
            'clustered': clustered,
            'unclustered': total - clustered,
            'total_clusters': num_clusters,
        }

    def get_article_id_by_url(self, url: str) -> Optional[int]:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute('SELECT id FROM articles WHERE url=%s', (url,))
                row = cur.fetchone()
            self._release(conn)
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error fetching article id by url: {e}")
            return None

    # ── custom sources ─────────────────────────────────────────────────────

    def add_custom_source(self, name: str, url: str) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO custom_sources (name, url) VALUES (%s, %s) ON CONFLICT (url) DO NOTHING",
                    (name, url)
                )
                saved = cur.rowcount > 0
            conn.commit()
            self._release(conn)
            return saved
        except Exception as e:
            logger.error(f"Error adding custom source: {e}")
            return False

    def get_custom_sources(self) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, url, added_at, last_scraped FROM custom_sources WHERE active=1"
            )
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def delete_custom_source(self, source_id: int) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM custom_sources WHERE id=%s", (source_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            self._release(conn)
            return deleted
        except Exception as e:
            logger.error(f"Error deleting custom source: {e}")
            return False

    def touch_custom_source(self, source_id: int):
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE custom_sources SET last_scraped=CURRENT_TIMESTAMP WHERE id=%s",
                (source_id,)
            )
        conn.commit()
        self._release(conn)

    def reset_duplicate_bodies(self, min_occurrences: int = 2) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute('''
                SELECT a.id FROM articles a
                JOIN (
                    SELECT source, body
                    FROM articles
                    WHERE body IS NOT NULL AND body != '' AND LENGTH(body) >= 80
                    GROUP BY source, body
                    HAVING COUNT(*) >= %s
                ) dups ON a.source = dups.source AND a.body = dups.body
            ''', (min_occurrences,))
            ids = [r[0] for r in cur.fetchall()]
            if ids:
                cur.execute(
                    f"UPDATE articles SET body='', body_cleaned=NULL, embedding=NULL, cluster_id=NULL "
                    f"WHERE id = ANY(%s)",
                    (ids,)
                )
        conn.commit()
        self._release(conn)
        return len(ids)

    def get_all_titles(self) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, title FROM articles")
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def get_all_cluster_labels(self) -> List[Dict]:
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT cluster_id, event_title FROM cluster_labels")
            rows = cur.fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    # ── legacy helpers ─────────────────────────────────────────────────────

    def get_recent_articles(self, limit=10):
        return self.get_articles(limit=limit)

    def get_articles_by_source(self, source, limit=20):
        return self.get_articles(source=source, limit=limit)

    def get_articles_by_category(self, category, limit=20):
        return self.get_articles(category=category, limit=limit)

    def clear_old_articles(self, days=30):
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM articles WHERE scraped_at < NOW() - INTERVAL '%s days'" % int(days)
                )
                deleted = cur.rowcount
            conn.commit()
            self._release(conn)
            return deleted
        except Exception as e:
            logger.error(f"Error clearing old articles: {e}")
            return 0
