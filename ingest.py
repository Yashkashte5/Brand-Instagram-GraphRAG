from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import lancedb
import pyarrow as pa
from apify_client import ApifyClient
from dotenv import load_dotenv
from groq import Groq
import instructor
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

try:
    from graspologic.partition import leiden
    _LEIDEN_AVAILABLE = True
except ImportError:
    leiden = None           
    _LEIDEN_AVAILABLE = False

load_dotenv()

DATA_DIR  = Path("data")
ACTOR_ID  = "shu8hvrXbJbY3Eb9W"
DAYS_BACK = 60
MAX_POSTS  = 200

EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"
BGE_PREFIX          = "Represent this sentence: "
BGE_DOCUMENT_PREFIX = BGE_PREFIX
BGE_QUERY_PREFIX    = BGE_PREFIX

LLM_MODEL = "llama-3.3-70b-versatile"

_POSTS_TABLE    = "posts"
_ENTITIES_TABLE = "entities"

_embedding_model: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def _account_dir(username: str) -> Path:
    return DATA_DIR / username.lower().lstrip("@")

def raw_path(username: str) -> Path:
    return _account_dir(username) / "raw.json"

def processed_path(username: str) -> Path:
    return _account_dir(username) / "processed.json"

def lancedb_path(username: str) -> Path:
    return _account_dir(username) / "graph_store" / "lancedb"

def graph_path(username: str) -> Path:
    return _account_dir(username) / "graph_store" / "graph.gpickle"

def communities_path(username: str) -> Path:
    return _account_dir(username) / "graph_store" / "communities.json"

def _parse_ts(raw) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None

def _caption(item: dict) -> str:
    cap = item.get("caption") or item.get("edge_media_to_caption", {})
    if isinstance(cap, str):
        return cap
    if isinstance(cap, dict):
        edges = cap.get("edges", [])
        return edges[0].get("node", {}).get("text", "") if edges else ""
    return ""

def _media_type(item: dict) -> str:
    raw = item.get("type") or item.get("media_type") or "image"
    if isinstance(raw, int):
        return {1: "image", 2: "video", 8: "carousel"}.get(raw, "image")
    return {
        "image": "image", "graphimage": "image",
        "video": "video", "graphvideo": "video",
        "sidecar": "carousel", "graphsidecar": "carousel", "carousel": "carousel",
    }.get(str(raw).lower(), "image")

def _post_id(item: dict) -> str:
    return str(item.get("shortCode") or item.get("shortcode")
               or item.get("id") or item.get("pk") or "")

def _extract_hashtags(caption: str, apify_tags: list) -> list[str]:
    tags = set()
    for h in (apify_tags or []):
        if isinstance(h, str):
            tags.add(h.lstrip("#").lower())
    for t in re.findall(r"#(\w+)", caption):
        tags.add(t.lower())
    return sorted(tags)

def _extract_mentions(caption: str, apify_mentions: list, username: str) -> list[str]:
    mentions = set()
    for m in (apify_mentions or []):
        if isinstance(m, str):
            mentions.add(m.lstrip("@").lower())
    for m in re.findall(r"@(\w+)", caption):
        if m.lower() != username.lower():
            mentions.add(m.lower())
    return sorted(mentions)

class EntityItem(BaseModel):
    id:   str = Field(..., description="lowercase_snake_case unique slug, max 40 chars")
    type: Literal["Brand", "Product", "Person", "Event",
                  "Topic", "Emotion", "Location", "Campaign"]
    name: str = Field(..., max_length=80)

    @field_validator("id")
    @classmethod
    def clean_id(cls, v: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", v.lower())[:40].strip("_") or "entity"

class RelationshipItem(BaseModel):
    source:   str = Field(..., description="entity id from entities list")
    target:   str = Field(..., description="entity id from entities list")
    relation: Literal["PROMOTES", "COLLABORATES_WITH", "TARGETS",
                      "EVOKES", "LOCATED_AT", "PART_OF", "FEATURES", "RELATED_TO"]

class PostExtraction(BaseModel):
    entities:         list[EntityItem]       = Field(default_factory=list)
    relationships:    list[RelationshipItem] = Field(default_factory=list)
    content_type:     Literal["product_launch", "lifestyle", "educational",
                               "promotional", "behind_the_scenes",
                               "user_generated", "event", "other"] = "other"
    sentiment:        Literal["positive", "neutral", "negative"] = "neutral"
    audience_signals: list[str] = Field(default_factory=list)

LLM_SYSTEM = """You are an Instagram content analyst.
Given a post caption, extract entities and relationships, then classify the content.

Focus on:
- Entities: Brands, Products, People, Events, Topics, Emotions, Locations, Campaigns
- Semantic relationships between entities
- content_type classification, sentiment, audience_signals

Rules:
- entity id: lowercase_snake_case, max 40 chars
- extract 2-8 entities
- extract 1-6 relationships referencing entity ids
- audience_signals: 2-5 short phrases describing the target audience
"""

def _get_groq_instructor():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")
    return instructor.from_groq(Groq(api_key=key), mode=instructor.Mode.JSON)

def _extract_entities_batch(posts: list[dict]) -> dict[str, dict]:
    client    = _get_groq_instructor()
    results:  dict[str, dict] = {}
    captioned = [p for p in posts if (p.get("caption") or "").strip()]

    print(f"  Groq extraction on {len(captioned)} posts...")

    for i, post in enumerate(captioned):
        caption = (post["caption"] or "")[:800]
        prompt  = f"Caption:\n{caption}"
        try:
            extraction: PostExtraction = client.chat.completions.create(
                model=LLM_MODEL,
                response_model=PostExtraction,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_retries=3,
                temperature=0,
                max_tokens=700,
            )
            ents         = [e.model_dump() for e in extraction.entities]
            rels         = [r.model_dump() for r in extraction.relationships]
            content_type = extraction.content_type
            sentiment    = extraction.sentiment
            audience     = list(extraction.audience_signals)[:5]
        except Exception as exc:
            print(f"    ⚠  Groq failed for post {post['post_id']}: {exc}")
            ents, rels, content_type, sentiment, audience = [], [], "other", "neutral", []

        results[post["post_id"]] = {
            "entities":         ents,
            "relationships":    rels,
            "content_type":     content_type,
            "sentiment":        sentiment,
            "audience_signals": audience,
        }
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(captioned)} posts processed")

    for post in posts:
        if post["post_id"] not in results:
            results[post["post_id"]] = {
                "entities": [], "relationships": [],
                "content_type": "other", "sentiment": "neutral",
                "audience_signals": [],
            }
    return results

def _open_db(username: str) -> lancedb.DBConnection:
    p = lancedb_path(username)
    p.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(p))

def build_embeddings(posts: list[dict], username: str) -> None:
    db    = _open_db(username)
    model = _get_embedding_model()

    ids   = [p["post_id"] for p in posts]
    texts = [BGE_DOCUMENT_PREFIX + (p["caption"] or p["post_id"]) for p in posts]
    vecs  = model.encode(texts, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    dim   = int(vecs.shape[1])

    post_schema = pa.schema([
        pa.field("id",       pa.string()),
        pa.field("vector",   pa.list_(pa.float32(), dim)),
        pa.field("model",    pa.string()),
        pa.field("username", pa.string()),
    ])
    post_rows = [
        {"id": pid, "vector": vec.tolist(), "model": EMBEDDING_MODEL, "username": username}
        for pid, vec in zip(ids, vecs)
    ]
    if _POSTS_TABLE in db.table_names():
        db.drop_table(_POSTS_TABLE)
    db.create_table(_POSTS_TABLE, data=post_rows, schema=post_schema)

    seen_ents: dict[str, dict] = {}
    for post in posts:
        for ent in post.get("graph_entities", []):
            if ent["id"] not in seen_ents:
                seen_ents[ent["id"]] = ent

    if seen_ents:
        ent_ids   = list(seen_ents.keys())
        ent_texts = [
            BGE_DOCUMENT_PREFIX
            + f"{seen_ents[eid].get('type','Topic')}: {seen_ents[eid].get('name', eid)}"
            for eid in ent_ids
        ]
        ent_vecs = model.encode(ent_texts, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        ent_schema = pa.schema([
            pa.field("id",       pa.string()),
            pa.field("vector",   pa.list_(pa.float32(), dim)),
            pa.field("model",    pa.string()),
            pa.field("username", pa.string()),
        ])
        ent_rows = [
            {"id": f"entity:{eid}", "vector": vec.tolist(),
             "model": EMBEDDING_MODEL, "username": username}
            for eid, vec in zip(ent_ids, ent_vecs)
        ]
        if _ENTITIES_TABLE in db.table_names():
            db.drop_table(_ENTITIES_TABLE)
        db.create_table(_ENTITIES_TABLE, data=ent_rows, schema=ent_schema)

def search_embeddings(username: str, query: str, top_k: int = 10) -> list[dict]:
    db    = _open_db(username)
    model = _get_embedding_model()
    qvec  = model.encode(
        [BGE_QUERY_PREFIX + query],
        convert_to_numpy=True, normalize_embeddings=True,
    )[0].tolist()

    results: list[dict] = []
    for table_name in [_POSTS_TABLE, _ENTITIES_TABLE]:
        if table_name not in db.table_names():
            continue
        rows = (
            db.open_table(table_name)
              .search(qvec)
              .metric("cosine")
              .limit(top_k)
              .to_list()
        )
        for row in rows:
            results.append({
                "id":    row["id"],
                "score": float(1.0 - row.get("_distance", 1.0)),
            })

    seen:    set[str]   = set()
    deduped: list[dict] = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    return deduped[:top_k]

def embeddings_are_fresh(username: str, posts: list[dict]) -> bool:
    db = _open_db(username)
    if _POSTS_TABLE not in db.table_names():
        return False
    db_ids   = set(db.open_table(_POSTS_TABLE).to_pandas()["id"].tolist())
    post_ids = {p["post_id"] for p in posts}
    return db_ids == post_ids

def build_communities(posts: list[dict], username: str) -> list[dict]:
    cp = communities_path(username)
    cp.parent.mkdir(parents=True, exist_ok=True)

    if not _LEIDEN_AVAILABLE:
        cp.write_text(json.dumps([]))
        return []

    entity_info: dict[str, dict] = {}
    edge_list:   list[tuple]     = []

    for p in posts:
        for ent in p.get("graph_entities", []):
            eid = ent["id"]
            if eid not in entity_info:
                entity_info[eid] = {"name": ent.get("name", eid),
                                    "type": ent.get("type", "Topic")}
        for rel in p.get("graph_relationships", []):
            src = rel.get("source")
            tgt = rel.get("target")
            if src and tgt and src in entity_info and tgt in entity_info:
                edge_list.append((src, tgt))

    if len(entity_info) < 2 or not edge_list:
        cp.write_text(json.dumps([]))
        return []

    node_list = list(entity_info.keys())
    node_idx  = {n: i for i, n in enumerate(node_list)}
    idx_to_node = {i: n for n, i in node_idx.items()}

    weighted_edges = [
        (node_idx[s], node_idx[t], 1.0)
        for s, t in edge_list
        if s in node_idx and t in node_idx
    ]

    print(f"  Running Leiden on {len(node_list)} entities, {len(weighted_edges)} edges...")
    try:
        partition = leiden(weighted_edges, resolution=1.0, trials=3)
    except Exception as exc:
        print(f"  ⚠  Leiden failed: {exc}")
        cp.write_text(json.dumps([]))
        return []

    community_map: dict[int, list[str]] = {}

    for node_idx_val, comm_idx in partition.items():
        node_name = idx_to_node.get(node_idx_val)
        if node_name is None:
            continue
        community_map.setdefault(comm_idx, []).append(node_name)

    community_map = {k: v for k, v in community_map.items() if len(v) >= 2}

    groq_key    = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=groq_key) if groq_key else None

    SUMMARY_SYSTEM = (
        f"You are a brand analyst. The Instagram account @{username} has been "
        "analysed and its content entities grouped into semantic communities. "
        "Write a 2-3 sentence summary of what this community represents in terms "
        "of the brand's positioning, content themes, or audience targeting. "
        "Return only plain text — no bullet points, no JSON."
    )

    communities: list[dict] = []
    print(f"  Summarising {len(community_map)} communities...")

    for comm_idx, member_ids in sorted(community_map.items()):
        members_info = [
            {"id": f"entity:{eid}", "name": entity_info[eid]["name"],
             "type": entity_info[eid]["type"]}
            for eid in member_ids
        ]
        eid_freq = {}
        for p in posts:
            for ent in p.get("graph_entities", []):
                if ent["id"] in set(member_ids):
                    eid_freq[ent["id"]] = eid_freq.get(ent["id"], 0) + 1
        top_eids = [f"entity:{k}" for k, _ in
                    sorted(eid_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

        relevant = [p for p in posts if
                    {e["id"] for e in p.get("graph_entities", [])} & set(member_ids)]
        avg_likes = int(np.mean([p["like_count"] for p in relevant])) if relevant else 0

        summary = ""
        if groq_client:
            member_desc = ", ".join(
                f"{m['name']} ({m['type']})" for m in members_info[:12]
            )
            try:
                resp = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SUMMARY_SYSTEM},
                        {"role": "user",   "content":
                            f"Community entities: {member_desc}\n"
                            f"Account: @{username}"},
                    ],
                    temperature=0.4,
                    max_tokens=200,
                )
                summary = resp.choices[0].message.content.strip()
            except Exception as exc:
                print(f"    ⚠  Groq summary failed for community {comm_idx}: {exc}")

        communities.append({
            "community_id":   f"c_{comm_idx}",
            "members":        members_info,
            "size":           len(member_ids),
            "summary":        summary,
            "avg_likes":      avg_likes,
            "top_entity_ids": top_eids,
        })

    communities.sort(key=lambda c: c["size"], reverse=True)
    cp.write_text(json.dumps(communities, indent=2))
    print(f"  ✓ {len(communities)} communities detected and summarised")
    return communities

def load_communities(username: str) -> list[dict]:
    cp = communities_path(username)
    if not cp.exists():
        return []
    try:
        return json.loads(cp.read_text())
    except Exception:
        return []

def communities_are_fresh(username: str) -> bool:
    cp = communities_path(username)
    if not cp.exists():
        return False
    try:
        data = json.loads(cp.read_text())
        return isinstance(data, list)
    except Exception:
        return False

def normalise(item: dict, username: str) -> Optional[dict]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=DAYS_BACK)
    ts = _parse_ts(item.get("timestamp") or item.get("taken_at_timestamp"))
    if ts is None or ts < cutoff:
        return None
    pid = _post_id(item)
    if not pid:
        return None
    caption  = _caption(item)
    hashtags = _extract_hashtags(caption, item.get("hashtags") or [])
    mentions = _extract_mentions(caption, item.get("mentions") or [], username)
    return {
        "post_id":             pid,
        "username":            username.lower().lstrip("@"),
        "url":                 item.get("url") or f"https://www.instagram.com/p/{pid}/",
        "caption":             caption,
        "hashtags":            hashtags,
        "mentions":            mentions,
        "graph_entities":      [],
        "graph_relationships": [],
        "content_type":        "other",
        "sentiment":           "neutral",
        "audience_signals":    [],
        "like_count":    int(item.get("likesCount")    or item.get("edge_media_preview_like", {}).get("count", 0) or 0),
        "comment_count": int(item.get("commentsCount") or item.get("edge_media_to_comment",  {}).get("count", 0) or 0),
        "timestamp":     ts.isoformat(),
        "month":         ts.strftime("%Y-%m"),
        "media_type":    _media_type(item),
    }

def scrape_account(username: str, force: bool = False) -> list[dict]:
    username = username.lower().lstrip("@")
    pp = processed_path(username)
    rp = raw_path(username)

    if not force and pp.exists():
        try:
            posts = json.loads(pp.read_text())
            if posts:
                if not embeddings_are_fresh(username, posts):
                    print("LanceDB embeddings stale — rebuilding...")
                    build_embeddings(posts, username)
                if not communities_are_fresh(username):
                    print("Communities missing — running Leiden...")
                    build_communities(posts, username)
                return posts
        except json.JSONDecodeError:
            pass

    _account_dir(username).mkdir(parents=True, exist_ok=True)

    if not force and rp.exists():
        try:
            items = json.loads(rp.read_text())
        except json.JSONDecodeError:
            items = None
    else:
        items = None

    if items is None:
        apify_token = os.environ.get("APIFY_API_TOKEN")
        if not apify_token:
            raise EnvironmentError("APIFY_API_TOKEN not set in .env")
        client = ApifyClient(apify_token)
        run    = client.actor(ACTOR_ID).call(run_input={
            "directUrls":    [f"https://www.instagram.com/{username}/"],
            "resultsType":   "posts",
            "resultsLimit":  MAX_POSTS,
            "addParentData": False,
        })
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        rp.write_text(json.dumps(items, indent=2, default=str))

    processed, seen = [], set()
    for item in items:
        p = normalise(item, username)
        if p and p["post_id"] not in seen:
            seen.add(p["post_id"])
            processed.append(p)

    print(f"Running entity extraction on {len(processed)} posts...")
    extractions = _extract_entities_batch(processed)

    entity_count = rel_count = 0
    for p in processed:
        ex = extractions.get(p["post_id"], {})
        p["graph_entities"]      = ex.get("entities", [])
        p["graph_relationships"] = ex.get("relationships", [])
        p["content_type"]        = ex.get("content_type", "other")
        p["sentiment"]           = ex.get("sentiment", "neutral")
        p["audience_signals"]    = ex.get("audience_signals", [])
        entity_count += len(p["graph_entities"])
        rel_count    += len(p["graph_relationships"])

    print(f"  ✓ {entity_count} entities, {rel_count} relationships")

    pp.write_text(json.dumps(processed, indent=2))

    print("Building LanceDB embeddings...")
    build_embeddings(processed, username)

    print("Running community detection...")
    build_communities(processed, username)

    return processed

def get_cached_accounts() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and (d / "processed.json").exists()]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <instagram_username>")
        sys.exit(1)
    scrape_account(sys.argv[1])