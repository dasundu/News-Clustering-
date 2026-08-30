# searcher.py
"""
Claim-to-event retrieval: free-text news claim in, matching event clusters out.

Serves the external AI-agent component's question "is this news in your
database?" — a user describes some news, the agent forwards the text here,
and this returns the event clusters that cover it (with their cluster_ids)
so the agent can do whatever it likes with them.

Why this isn't RAG
------------------
RAG is retrieval *plus* generation. Only the retrieval half belongs here:
the consumer wants cluster ids, not prose, and keeping generation out means
nothing this module returns can be hallucinated. Academically the task is
claim matching / evidence retrieval (cf. CLEF CheckThat!), which is also
precisely what a fact-checking or bias-detection consumer needs.

Why hybrid retrieval
--------------------
Two channels with complementary blind spots, run over the same corpus:

  * dense (pgvector cosine over article embeddings) handles paraphrase —
    "central bank cut rates" matches "CBSL reduces policy rate" — but is
    weak on exact tokens like "Rs 200bn" or an unusual proper noun.
  * lexical (Postgres full-text search) is the exact opposite: precise on
    entities and numbers, blind to rewording.

Their rankings are combined with Reciprocal Rank Fusion (Cormack et al.
2009), chosen over a weighted score blend because it needs no score
normalisation and introduces no tuned weight — the two channels' scores
(cosine similarity vs ts_rank_cd) aren't on comparable scales, and rank
fusion sidesteps that entirely rather than papering over it.

Why results are clusters, not articles
--------------------------------------
Returning eight near-identical articles about one event would throw away
the thing this system exists to do. Articles are scored only as *evidence*;
the scores are aggregated up to their event cluster, and clusters are what
gets returned. A cluster answer also carries more information — it says
which outlets covered the story, not just that someone did.
"""

import logging
from typing import Dict, List, Optional

import embedding_provider
from database import ArticleDatabase
from config import (
    SEARCH_CANDIDATES, SEARCH_TOP_K, SEARCH_RRF_K,
    SEARCH_MIN_SCORE_HIGH, SEARCH_MIN_SCORE_LOW, SEARCH_AGGREGATION,
    SEARCH_CORROBORATION_RANK, SEARCH_MIN_SCORE_FLOOR,
)

logger = logging.getLogger(__name__)


def _rrf_fuse(channels: List[List[Dict]], k: int) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion: an article's fused score is the sum over channels
    of 1/(k + rank), rank being 1-based within that channel. Articles found
    by both channels are rewarded without either channel's raw scores having
    to be made commensurable. Returns {article_id: fused_score}.
    """
    fused: Dict[int, float] = {}
    for rows in channels:
        for rank, row in enumerate(rows, start=1):
            fused[row['id']] = fused.get(row['id'], 0.0) + 1.0 / (k + rank)
    return fused


class ClaimSearcher:
    def __init__(self, db: Optional[ArticleDatabase] = None):
        self.db = db or ArticleDatabase()

    def search(self, text: str, top_k: int = SEARCH_TOP_K) -> Dict:
        """
        Returns a graded result rather than a bare list:

          status='found'     best dense similarity >= SEARCH_MIN_SCORE_HIGH
          status='partial'   between the two thresholds — candidates are
                             returned but explicitly not asserted as a match
          status='not_found' below SEARCH_MIN_SCORE_LOW

        The middle tier exists because a single hard cutoff forces every
        near-miss into one of two misleading answers. Returning "nothing
        conclusive, but here are related events" keeps the judgement with the
        calling agent instead of hiding it behind this module's threshold.
        """
        text = (text or '').strip()
        if not text:
            return self._empty('not_found', text, 0.0)

        # ── the two retrieval channels ────────────────────────────────────
        query_vec = self._embed_query(text)
        dense = self.db.search_dense(query_vec, limit=SEARCH_CANDIDATES)
        lexical = self.db.search_lexical(text, limit=SEARCH_CANDIDATES)

        if not dense and not lexical:
            return self._empty('not_found', text, 0.0)

        # ── fuse, then collapse articles into their clusters ──────────────
        fused = _rrf_fuse([dense, lexical], SEARCH_RRF_K)

        # Dense cosine per article, kept separate from the RRF score: RRF is
        # rank-derived and not comparable across queries, so it decides
        # *ordering* while cosine — bounded and interpretable — decides
        # confidence and thresholding.
        dense_sim = {r['id']: float(r['score']) for r in dense}
        by_id = {r['id']: r for r in list(lexical) + list(dense)}

        clusters: Dict[int, Dict] = {}
        for article_id, rrf_score in fused.items():
            row = by_id[article_id]
            cid = row['cluster_id']
            entry = clusters.setdefault(cid, {'rrf': 0.0, 'sims': [], 'articles': []})
            # max, not sum: summing RRF across a cluster's members would rank
            # by how many articles an event happens to have rather than how
            # well any of them matches — the same size bias that rules sum()
            # out for the confidence aggregation below.
            entry['rrf'] = max(entry['rrf'], rrf_score)
            sim = dense_sim.get(article_id)
            if sim is not None:
                # Articles found only by the lexical channel have no cosine
                # here (they fell outside the dense top-N). They still count
                # as evidence via RRF, but injecting 0.0 would actively drag
                # a mean-based confidence down, so they're simply omitted.
                entry['sims'].append(sim)
            entry['articles'].append({
                'id': article_id,
                'title': row['title'],
                'source': row['source'],
                'url': row['url'],
                'score': round(sim, 4) if sim is not None else None,
            })

        for entry in clusters.values():
            entry['confidence'] = self._aggregate(entry['sims'])
            entry['articles'].sort(key=lambda a: (a['score'] is not None, a['score']), reverse=True)

        # Two-step: RRF (fused evidence from both channels) decides *which*
        # clusters make the cut, then the returned set is ordered by the
        # interpretable confidence so the caller's first result is also the
        # most confident one — and best_score always describes matches[0]
        # rather than something further down the list.
        selected = sorted(clusters.items(), key=lambda kv: kv[1]['rrf'], reverse=True)[:top_k]

        # A cluster ranked near the top by BOTH channels has been found twice
        # by methods with opposite blind spots. That outranks a cluster only
        # the vector channel liked, even one with a slightly higher cosine —
        # measured on "Bandaranaike", where the dense channel's favourite was
        # an unrelated parliament story at 0.808 while the cluster both
        # channels agreed on sat second.
        for cid, entry in selected:
            entry['corroborated'] = self._corroborated(cid, dense, lexical)

        ranked = sorted(
            selected,
            key=lambda kv: (kv[1]['corroborated'], kv[1]['confidence']),
            reverse=True,
        )
        best_score = ranked[0][1]['confidence'] if ranked else 0.0
        corroborated = bool(ranked) and ranked[0][1]['corroborated']

        high = SEARCH_MIN_SCORE_LOW if corroborated else SEARCH_MIN_SCORE_HIGH
        low = SEARCH_MIN_SCORE_FLOOR if corroborated else SEARCH_MIN_SCORE_LOW
        status = (
            'found' if best_score >= high
            else 'partial' if best_score >= low
            else 'not_found'
        )

        if status == 'not_found':
            # Worth keeping: a coverage-gap signal, and a growing record of
            # real retrieval failures to analyse.
            self.db.log_unmatched_query(text, best_score)
            return self._empty('not_found', text, best_score)

        matches = self._decorate(ranked)
        return {
            'status': status,
            'query': text,
            'best_score': round(best_score, 4),
            'matches': matches,
            # Topic roll-up of whatever matched, so a caller that only cares
            # about the broader ongoing story doesn't have to group the
            # events itself.
            'matched_topics': self._topics_of(matches),
            'interpretation': None if status == 'found' else _PARTIAL_NOTE,
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> List[float]:
        """
        The query must be preprocessed *exactly* as documents were, or it
        lands somewhere else in the embedding space and every similarity is
        subtly wrong. embedder._build_text repeats the title and strips
        outlet names; a claim has no title/body split, so it plays the role
        of the title — which is also the part of a document embedding that
        dominates, making this claim-to-headline matching.
        """
        from embedder import ArticleEmbedder
        prepared = ArticleEmbedder._build_text({'title': text, 'body_cleaned': ''})
        return embedding_provider.encode([prepared], task='query')[0].tolist()

    @staticmethod
    def _corroborated(cluster_id: int, dense: List[Dict],
                      lexical: List[Dict]) -> bool:
        """
        True when this cluster is ranked inside the top
        SEARCH_CORROBORATION_RANK of BOTH channels.

        Rank, not score: the two channels' scores are not on comparable
        scales, which is the same reason fusion uses RRF. A cluster's rank in
        a channel is the best rank of any of its articles there.
        """
        def best_rank(rows: List[Dict]) -> int:
            for rank, row in enumerate(rows, start=1):
                if row['cluster_id'] == cluster_id:
                    return rank
            return 10 ** 6
        return (best_rank(dense) <= SEARCH_CORROBORATION_RANK
                and best_rank(lexical) <= SEARCH_CORROBORATION_RANK)

    @staticmethod
    def _aggregate(sims: List[float]) -> float:
        """
        Article evidence -> cluster confidence. 'max' treats one strongly
        matching article as sufficient proof the event is present; 'mean3'
        averages the top three, rewarding corroboration across outlets at the
        cost of penalising events only one outlet has covered so far. Sum is
        deliberately not an option — it would rank purely by cluster size.
        """
        if not sims:
            return 0.0
        if SEARCH_AGGREGATION == 'mean3':
            top = sorted(sims, reverse=True)[:3]
            return float(sum(top) / len(top))
        return float(max(sims))

    def _decorate(self, ranked: List) -> List[Dict]:
        """Attach event title, topic, outlets — the same shape /clusters
        returns, so a search hit is interchangeable with a feed item."""
        summaries = self.db.get_cluster_summaries([cid for cid, _ in ranked])
        matches = []
        for cid, entry in ranked:
            s = summaries.get(cid, {})
            matches.append({
                'cluster_id': cid,
                'event_title': s.get('event_title') or '',
                'confidence': round(entry['confidence'], 4),
                'topic_id': s.get('topic_id'),
                'topic_title': s.get('topic_title'),
                'category': s.get('category'),
                'article_count': s.get('article_count', len(entry['articles'])),
                'sources': s.get('sources', []),
                'latest_at': s.get('latest_at'),
                'matched_articles': entry['articles'][:5],
            })
        return matches

    @staticmethod
    def _topics_of(matches: List[Dict]) -> List[Dict]:
        """Distinct topics across the matched events, each scored by its
        best-matching member event."""
        topics: Dict[int, Dict] = {}
        for m in matches:
            tid = m.get('topic_id')
            if tid is None:
                continue
            existing = topics.get(tid)
            if existing is None or m['confidence'] > existing['confidence']:
                topics[tid] = {
                    'topic_id': tid,
                    'topic_title': m.get('topic_title') or '',
                    'confidence': m['confidence'],
                }
        return sorted(topics.values(), key=lambda t: t['confidence'], reverse=True)

    @staticmethod
    def _empty(status: str, query: str, best_score: float) -> Dict:
        return {
            'status': status,
            'query': query,
            'best_score': round(best_score, 4),
            'matches': [],
            'matched_topics': [],
            'interpretation': _NOT_FOUND_NOTE,
        }


# Stated explicitly in the payload rather than left to the caller to infer.
# A consumer feeding a fact-checking system must not read "we have no
# coverage of this" as "this claim is false" — those are different claims,
# and conflating them would be a correctness bug, not a wording nicety.
_NOT_FOUND_NOTE = 'no_matching_coverage_found_in_indexed_sources'
_PARTIAL_NOTE = 'related_coverage_only_not_a_confirmed_match'


if __name__ == '__main__':
    import json
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    query = ' '.join(sys.argv[1:]) or 'Sri Lanka IMF loan approval'
    print(json.dumps(ClaimSearcher().search(query), indent=2, ensure_ascii=False))
