from __future__ import annotations

import json
import pickle
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from ingest import (
    processed_path, graph_path,
    build_embeddings, search_embeddings, embeddings_are_fresh,
    load_communities,
    DAYS_BACK,
)

_GLOBAL_KEYWORDS = {
    "overall", "brand", "identity", "positioning", "strategy", "about",
    "overview", "summarise", "summarize", "tell me about", "what is",
    "what does", "who is", "describe", "in general", "broadly", "theme",
    "themes", "narrative", "voice", "persona", "values", "mission",
    "personality", "reputation", "image", "perception",
}

def _is_global_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _GLOBAL_KEYWORDS)

def build_graph(username: str, force_rebuild: bool = False) -> nx.DiGraph:
    username = username.lower().lstrip("@")
    gp = graph_path(username)
    gp.parent.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and gp.exists():
        try:
            return _load(username)
        except Exception:
            gp.unlink(missing_ok=True)

    posts = _posts(username)
    G     = nx.DiGraph()
    G.add_node(username, node_type="brand", label=f"@{username}")

    for p in posts:
        pid = p["post_id"]
        G.add_node(pid,
                   node_type     = "post",
                   label         = f"Post:{pid[:8]}",
                   like_count    = p["like_count"],
                   comment_count = p["comment_count"],
                   media_type    = p["media_type"],
                   timestamp     = p["timestamp"],
                   month         = p["month"],
                   caption       = (p["caption"] or "")[:300],
                   url           = p["url"],
                   content_type  = p.get("content_type", "other"),
                   sentiment     = p.get("sentiment", "neutral"))
        G.add_edge(username, pid, relation="POSTED")

        ent_id_map: dict[str, str] = {}
        for ent in p.get("graph_entities", []):
            eid   = f"entity:{ent['id']}"
            etype = ent.get("type", "Topic")
            ename = ent.get("name", ent["id"])
            ent_id_map[ent["id"]] = eid
            if not G.has_node(eid):
                G.add_node(eid, node_type="entity", entity_type=etype,
                           label=ename, raw_id=ent["id"])
            G.add_edge(pid, eid, relation="CONTAINS_ENTITY")

        for rel in p.get("graph_relationships", []):
            src = ent_id_map.get(rel.get("source"))
            tgt = ent_id_map.get(rel.get("target"))
            if src and tgt and G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, relation=rel.get("relation", "RELATED_TO"))

        for signal in p.get("audience_signals", []):
            sig_id = f"audience:{signal.lower().replace(' ', '_')[:40]}"
            if not G.has_node(sig_id):
                G.add_node(sig_id, node_type="audience", label=signal)
            G.add_edge(pid, sig_id, relation="TARGETS_AUDIENCE")

        for tag in p.get("hashtags", []):
            tid = f"#{tag}"
            if not G.has_node(tid):
                G.add_node(tid, node_type="hashtag", label=tid)
            G.add_edge(pid, tid, relation="HAS_HASHTAG")

        for mention in p.get("mentions", []):
            mid = f"@{mention}"
            if not G.has_node(mid):
                G.add_node(mid, node_type="mention", label=mid)
            G.add_edge(pid, mid, relation="MENTIONS")

        month_id = f"month:{p['month']}"
        if not G.has_node(month_id):
            G.add_node(month_id, node_type="month", label=p["month"])
        G.add_edge(pid, month_id, relation="BELONGS_TO")

        mt_id = f"media:{p['media_type']}"
        if not G.has_node(mt_id):
            G.add_node(mt_id, node_type="media_type", label=p["media_type"].upper())
        G.add_edge(pid, mt_id, relation="IS_TYPE")

        ct_id = f"content_type:{p.get('content_type', 'other')}"
        if not G.has_node(ct_id):
            G.add_node(ct_id, node_type="content_type",
                       label=p.get("content_type", "other").replace("_", " ").title())
        G.add_edge(pid, ct_id, relation="HAS_CONTENT_TYPE")

        sent_id = f"sentiment:{p.get('sentiment', 'neutral')}"
        if not G.has_node(sent_id):
            G.add_node(sent_id, node_type="sentiment",
                       label=p.get("sentiment", "neutral").title())
        G.add_edge(pid, sent_id, relation="HAS_SENTIMENT")

    for comm in load_communities(username):
        cid = f"community:{comm['community_id']}"
        G.add_node(cid, node_type="community",
                   community_id = comm["community_id"],
                   summary      = comm.get("summary", ""),
                   size         = comm["size"],
                   avg_likes    = comm.get("avg_likes", 0),
                   label        = f"Community {comm['community_id']}")
        for member in comm.get("members", []):
            if G.has_node(member["id"]):
                G.add_edge(cid, member["id"], relation="HAS_MEMBER")

    with open(gp, "wb") as f:
        pickle.dump(G, f)
    return G

def _load(username: str) -> nx.DiGraph:
    with open(graph_path(username), "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Invalid graph file")
    return G

def _posts(username: str) -> list[dict]:
    pp = processed_path(username)
    if not pp.exists():
        raise FileNotFoundError(f"No data for @{username}. Run scrape_account() first.")
    return json.loads(pp.read_text())

def query_graph(
    username:      str,
    query:         str,
    top_k_seeds:   int = 5,
    top_k_nodes:   int = 30,
    include_posts: int = 8,
    bfs_depth:     int = 2,
) -> dict[str, Any]:
    if _is_global_query(query):
        return _query_global(username, query)
    return _query_local(username, query, top_k_seeds, top_k_nodes, include_posts, bfs_depth)

def _query_global(username: str, query: str) -> dict[str, Any]:
    communities = load_communities(username)
    posts       = _posts(username)

    if not communities:
        return _query_local(username, query)

    from ingest import _get_embedding_model, BGE_QUERY_PREFIX, BGE_DOCUMENT_PREFIX
    model = _get_embedding_model()
    qvec  = model.encode(
        [BGE_QUERY_PREFIX + query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    scored: list[tuple[float, dict]] = []
    for comm in communities:
        summary = comm.get("summary") or ", ".join(m["name"] for m in comm["members"][:8])
        svec = model.encode(
            [BGE_DOCUMENT_PREFIX + summary], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scored.append((float(np.dot(qvec, svec)), comm))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_communities = [c for _, c in scored[:6]]

    highlight_ids: list[str] = []
    for comm in top_communities[:2]:
        highlight_ids.extend(comm.get("top_entity_ids", [])[:3])
    highlight_ids = list(dict.fromkeys(highlight_ids))[:10]

    return {
        "query":            query,
        "username":         username,
        "retrieval_method": "global_community_summaries",
        "retrieval_mode":   "global",
        "highlight_ids":    highlight_ids,
        "seed_nodes":       [],
        "communities":      top_communities,
        "subgraph": {
            "entities": [], "posts": [], "context_nodes": [],
            "relationships": [], "total_nodes": len(top_communities), "total_edges": 0,
        },
        "summary_stats": _quick_stats(posts),
    }

DEPTH_DECAY = 0.6

def _engagement_weight(node: str, G: nx.DiGraph, max_likes: int) -> float:
    data = G.nodes[node]
    if data.get("node_type") == "post":
        return 0.5 + 0.5 * (data.get("like_count", 0) / max(max_likes, 1))
    return 0.5

def _query_local(
    username:      str,
    query:         str,
    top_k_seeds:   int = 5,
    top_k_nodes:   int = 30,
    include_posts: int = 8,
    bfs_depth:     int = 2,
) -> dict[str, Any]:
    G        = build_graph(username)
    posts    = _posts(username)
    if not posts:
        return {"error": f"No posts for @{username}"}

    post_map  = {p["post_id"]: p for p in posts}
    max_likes = max((p["like_count"] for p in posts), default=1)

    raw_seeds = search_embeddings(username, query, top_k=top_k_seeds * 2)
    if not raw_seeds:
        build_embeddings(posts, username)
        raw_seeds = search_embeddings(username, query, top_k=top_k_seeds * 2)

    seed_scores: dict[str, float] = {
        r["id"]: r["score"] for r in raw_seeds if r["id"] in G.nodes
    }
    if not seed_scores:
        return {"error": "No seed nodes found. Re-scrape the account."}

    top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_seeds]

    node_scores: dict[str, float] = {}
    for seed_node, seed_sim in top_seeds:
        frontier: list[tuple[str, int, float]] = [(seed_node, 0, seed_sim)]
        local_visited: set[str] = set()
        while frontier:
            node, depth, score = frontier.pop(0)
            if node in local_visited or depth > bfs_depth:
                continue
            local_visited.add(node)
            eng   = _engagement_weight(node, G, max_likes)
            final = score * eng * (DEPTH_DECAY ** depth)
            node_scores[node] = max(node_scores.get(node, 0.0), final)
            for nb in list(G.successors(node)) + list(G.predecessors(node)):
                if nb not in local_visited:
                    frontier.append((nb, depth + 1, score * DEPTH_DECAY))

    ranked    = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    selected: set[str] = {username}
    post_count = 0
    for node, _ in ranked:
        if len(selected) >= top_k_nodes:
            break
        ntype = G.nodes[node].get("node_type", "")
        if ntype == "post":
            if post_count >= include_posts:
                continue
            post_count += 1
        selected.add(node)

    subgraph = G.subgraph(selected)

    entity_nodes: list[dict] = []
    post_nodes:   list[dict] = []
    other_nodes:  list[dict] = []

    for node in subgraph.nodes:
        data  = dict(subgraph.nodes[node])
        ntype = data.get("node_type", "")
        entry = {
            "node_id":          node,
            "bfs_score":        round(node_scores.get(node, 0.0), 6),
            "query_similarity": round(seed_scores.get(node, 0.0), 4),
            **data,
        }
        if ntype == "entity":
            pred_posts = [n for n in G.predecessors(node)
                          if G.nodes[n].get("node_type") == "post"]
            entry["used_in_posts"]   = len(pred_posts)
            entry["avg_likes_posts"] = (
                int(np.mean([post_map[n]["like_count"]
                             for n in pred_posts if n in post_map]))
                if pred_posts else 0
            )
            entity_nodes.append(entry)
        elif ntype == "post":
            p = post_map.get(node, {})
            entry.update({
                "caption_preview":  (p.get("caption") or "")[:200],
                "hashtags":         p.get("hashtags", [])[:10],
                "graph_entities":   p.get("graph_entities", []),
                "audience_signals": p.get("audience_signals", []),
                "content_type":     p.get("content_type", "other"),
                "sentiment":        p.get("sentiment", "neutral"),
            })
            post_nodes.append(entry)
        else:
            other_nodes.append(entry)

    highlight_ids = [
        node for node, _ in ranked
        if G.nodes[node].get("node_type") in ("post", "entity") and node in selected
    ][:include_posts]

    relationships = [
        {"source": s, "target": t, "relation": d.get("relation", "RELATED_TO")}
        for s, t, d in subgraph.edges(data=True)
    ]

    return {
        "query":            query,
        "username":         username,
        "retrieval_method": "weighted_bfs_lancedb",
        "retrieval_mode":   "local",
        "highlight_ids":    highlight_ids,
        "seed_nodes":       [{"node": n, "similarity": round(s, 4)} for n, s in top_seeds],
        "subgraph": {
            "entities":      entity_nodes,
            "posts":         sorted(post_nodes, key=lambda x: x.get("like_count", 0), reverse=True),
            "context_nodes": other_nodes,
            "relationships": relationships,
            "total_nodes":   len(selected),
            "total_edges":   subgraph.number_of_edges(),
        },
        "summary_stats": _quick_stats(posts),
    }

def _count_field(posts: list[dict], field: str) -> dict:
    counts: dict = defaultdict(int)
    for p in posts:
        counts[p.get(field, "unknown")] += 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

def get_engagement_summary(username: str) -> dict[str, Any]:
    posts = _posts(username)
    if not posts:
        return {"error": "No posts found."}
    likes    = [p["like_count"]    for p in posts]
    comments = [p["comment_count"] for p in posts]
    best     = max(posts, key=lambda p: p["like_count"])
    media:   dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    monthly: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    for p in posts:
        media[p["media_type"]]["count"]    += 1
        media[p["media_type"]]["likes"]    += p["like_count"]
        media[p["media_type"]]["comments"] += p["comment_count"]
        monthly[p["month"]]["count"]       += 1
        monthly[p["month"]]["likes"]       += p["like_count"]
        monthly[p["month"]]["comments"]    += p["comment_count"]
    s = sorted(likes); n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2
    return {
        "username":       username,
        "total_posts":    len(posts),
        "total_likes":    sum(likes),
        "total_comments": sum(comments),
        "avg_likes":      sum(likes) // len(likes),
        "median_likes":   median,
        "avg_comments":   sum(comments) // len(comments),
        "best_post": {
            "post_id":        best["post_id"],
            "url":            best["url"],
            "like_count":     best["like_count"],
            "comment_count":  best["comment_count"],
            "media_type":     best["media_type"],
            "content_type":   best.get("content_type", "other"),
            "sentiment":      best.get("sentiment", "neutral"),
            "caption_preview":(best["caption"] or "")[:140],
        },
        "media_breakdown": {
            mt: {"post_count": v["count"],
                 "avg_likes":    v["likes"] // v["count"],
                 "avg_comments": v["comments"] // v["count"]}
            for mt, v in media.items()
        },
        "monthly_trend": {
            m: {"post_count": v["count"],
                "avg_likes":    v["likes"] // v["count"],
                "avg_comments": v["comments"] // v["count"]}
            for m, v in sorted(monthly.items())
        },
        "content_type_breakdown": _count_field(posts, "content_type"),
        "sentiment_breakdown":    _count_field(posts, "sentiment"),
    }

def get_hashtag_analysis(username: str) -> dict[str, Any]:
    posts = _posts(username)
    stats: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    for p in posts:
        for tag in p["hashtags"]:
            stats[tag]["count"]    += 1
            stats[tag]["likes"]    += p["like_count"]
            stats[tag]["comments"] += p["comment_count"]
    rows = [
        {"hashtag": f"#{t}", "frequency": v["count"],
         "avg_likes": v["likes"] // v["count"],
         "avg_comments": v["comments"] // v["count"]}
        for t, v in stats.items()
    ]
    rows.sort(key=lambda x: x["frequency"], reverse=True)
    return {
        "username":              username,
        "total_unique_hashtags": len(rows),
        "top_by_frequency":      rows[:25],
        "top_by_avg_likes":      sorted(rows, key=lambda x: x["avg_likes"], reverse=True)[:10],
    }

def get_monthly_breakdown(username: str) -> dict[str, Any]:
    posts = _posts(username)
    monthly: dict = defaultdict(
        lambda: {"count": 0, "likes": 0, "comments": 0, "media": defaultdict(int)}
    )
    for p in posts:
        m = p["month"]
        monthly[m]["count"]    += 1
        monthly[m]["likes"]    += p["like_count"]
        monthly[m]["comments"] += p["comment_count"]
        monthly[m]["media"][p["media_type"]] += 1
    return {
        m: {"posts": v["count"], "total_likes": v["likes"],
            "avg_likes": v["likes"] // v["count"],
            "avg_comments": v["comments"] // v["count"],
            "media_mix": dict(v["media"])}
        for m, v in sorted(monthly.items())
    }

def get_entity_analysis(username: str) -> dict[str, Any]:
    posts = _posts(username)
    entity_stats:    dict = defaultdict(lambda: {"count": 0, "likes": 0, "types": set()})
    relation_counts: dict = defaultdict(int)
    for p in posts:
        for ent in p.get("graph_entities", []):
            name = ent.get("name", ent["id"])
            entity_stats[name]["count"] += 1
            entity_stats[name]["likes"] += p["like_count"]
            entity_stats[name]["types"].add(ent.get("type", "Topic"))
        for rel in p.get("graph_relationships", []):
            relation_counts[rel.get("relation", "UNKNOWN")] += 1
    rows = [
        {"entity": name, "type": list(v["types"])[0] if v["types"] else "Topic",
         "frequency": v["count"], "avg_likes": v["likes"] // v["count"] if v["count"] else 0}
        for name, v in entity_stats.items()
    ]
    rows.sort(key=lambda x: x["frequency"], reverse=True)
    return {
        "username":       username,
        "total_entities": len(rows),
        "top_entities":   rows[:20],
        "relation_types": dict(sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)),
        "content_types":  _count_field(posts, "content_type"),
        "sentiment_mix":  _count_field(posts, "sentiment"),
    }

def get_graph_context(username: str) -> dict[str, Any]:
    posts = _posts(username)
    if not posts:
        return {"error": f"No posts for @{username}."}
    return {
        "username":    username,
        "period":      f"last {DAYS_BACK} days",
        "engagement":  get_engagement_summary(username),
        "hashtags":    get_hashtag_analysis(username),
        "monthly":     get_monthly_breakdown(username),
        "entities":    get_entity_analysis(username),
        "communities": load_communities(username),
        "posts": [
            {
                "post_id":          p["post_id"],
                "url":              p["url"],
                "caption":          p["caption"],
                "hashtags":         p["hashtags"],
                "graph_entities":   p.get("graph_entities", []),
                "audience_signals": p.get("audience_signals", []),
                "content_type":     p.get("content_type", "other"),
                "sentiment":        p.get("sentiment", "neutral"),
                "like_count":       p["like_count"],
                "comment_count":    p["comment_count"],
                "media_type":       p["media_type"],
                "month":            p["month"],
                "timestamp":        p["timestamp"],
            }
            for p in posts
        ],
    }

def get_comparison_context(usernames: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparison_period": f"last {DAYS_BACK} days",
        "accounts":          {},
        "shared_hashtags":   [],
        "shared_entities":   [],
    }
    hashtag_sets: list[set] = []
    entity_sets:  list[set] = []

    for username in usernames:
        try:
            ctx = get_graph_context(username)
            result["accounts"][username] = ctx
            tags = {r["hashtag"].lstrip("#")
                    for r in ctx["hashtags"].get("top_by_frequency", [])}
            ents = {e["entity"] for e in ctx["entities"].get("top_entities", [])}
            hashtag_sets.append(tags)
            entity_sets.append(ents)
        except Exception as e:
            result["accounts"][username] = {"error": str(e)}

    if len(hashtag_sets) >= 2:
        shared = hashtag_sets[0]
        for s in hashtag_sets[1:]:
            shared &= s
        result["shared_hashtags"] = sorted(shared)

    if len(entity_sets) >= 2:
        shared_e = entity_sets[0]
        for s in entity_sets[1:]:
            shared_e &= s
        result["shared_entities"] = sorted(shared_e)

    return result

def _quick_stats(posts: list[dict]) -> dict:
    if not posts:
        return {}
    likes = [p["like_count"] for p in posts]
    return {"total_posts": len(posts), "avg_likes": int(np.mean(likes)), "max_likes": max(likes)}