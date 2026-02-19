from __future__ import annotations

import json
import os
import re
from groq import Groq
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Graph RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GROUP_STYLES = {
    "brand":        {"color": "#FF4500", "glow": "#FF4500", "size": 50},
    "post":         {"color": "#1E90FF", "glow": "#1E90FF", "size": None},
    "entity":       {"color": "#E040FB", "glow": "#E040FB", "size": 22},
    "hashtag":      {"color": "#00E676", "glow": "#00E676", "size": 14},
    "mention":      {"color": "#FF9800", "glow": "#FF9800", "size": 18},
    "audience":     {"color": "#FF6EC7", "glow": "#FF6EC7", "size": 16},
    "month":        {"color": "#FFD600", "glow": "#FFD600", "size": 24},
    "media_type":   {"color": "#26C6DA", "glow": "#26C6DA", "size": 20},
    "content_type": {"color": "#80DEEA", "glow": "#80DEEA", "size": 18},
    "sentiment":    {"color": "#A5D6A7", "glow": "#A5D6A7", "size": 16},
}

COMPARISON_BRAND_STYLES = {
    "brand_a":      {"color": "#FF4500", "glow": "#FF4500", "size": 50},
    "brand_b":      {"color": "#00B4FF", "glow": "#00B4FF", "size": 50},
    "brand_c":      {"color": "#00E676", "glow": "#00E676", "size": 50},
    "brand_d":      {"color": "#FFD600", "glow": "#FFD600", "size": 50},
    "post_a":       {"color": "#FF7043", "glow": "#FF7043", "size": None},
    "post_b":       {"color": "#29B6F6", "glow": "#29B6F6", "size": None},
    "post_c":       {"color": "#66BB6A", "glow": "#66BB6A", "size": None},
    "post_d":       {"color": "#FFF176", "glow": "#FFF176", "size": None},
    "entity":       {"color": "#E040FB", "glow": "#E040FB", "size": 22},
    "hashtag":      {"color": "#00E676", "glow": "#00E676", "size": 14},
    "mention":      {"color": "#FF9800", "glow": "#FF9800", "size": 18},
    "audience":     {"color": "#FF6EC7", "glow": "#FF6EC7", "size": 16},
    "month":        {"color": "#FFD600", "glow": "#FFD600", "size": 24},
    "media_type":   {"color": "#26C6DA", "glow": "#26C6DA", "size": 20},
    "content_type": {"color": "#80DEEA", "glow": "#80DEEA", "size": 18},
    "sentiment":    {"color": "#A5D6A7", "glow": "#A5D6A7", "size": 16},
}

EDGE_COLORS = {
    "POSTED":             "#FF450060",
    "CONTAINS_ENTITY":    "#E040FB60",
    "HAS_HASHTAG":        "#00E67650",
    "MENTIONS":           "#FF980060",
    "TARGETS_AUDIENCE":   "#FF6EC750",
    "BELONGS_TO":         "#FFD60050",
    "IS_TYPE":            "#26C6DA50",
    "HAS_CONTENT_TYPE":   "#80DEEA50",
    "HAS_SENTIMENT":      "#A5D6A750",
    "PROMOTES":           "#FF4500A0",
    "COLLABORATES_WITH":  "#00B4FFA0",
    "TARGETS":            "#FF6EC7A0",
    "EVOKES":             "#E040FBA0",
    "LOCATED_AT":         "#FFD600A0",
    "PART_OF":            "#26C6DAA0",
    "FEATURES":           "#00E676A0",
    "RELATED_TO":         "#ffffff30",
}

EDGE_WIDTHS = {
    "POSTED":             2.5,
    "CONTAINS_ENTITY":    2.0,
    "PROMOTES":           2.0,
    "COLLABORATES_WITH":  2.0,
    "MENTIONS":           1.8,
    "TARGETS":            1.8,
    "EVOKES":             1.8,
    "HAS_HASHTAG":        1.2,
    "TARGETS_AUDIENCE":   1.2,
    "BELONGS_TO":         1.0,
    "IS_TYPE":            1.0,
    "HAS_CONTENT_TYPE":   1.0,
    "HAS_SENTIMENT":      1.0,
    "RELATED_TO":         0.8,
}

from graph import build_graph, query_graph, get_cached_accounts, get_comparison_context
from ingest import scrape_account, processed_path, load_communities

GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
GROQ_CLIENT = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
GROQ_MODEL  = "llama-3.3-70b-versatile"

ANSWER_SYSTEM = """You are an expert Instagram brand strategist with access to a
knowledge graph built by Graph RAG over Instagram posts.

Three retrieval modes:
- GLOBAL: context is community summaries covering the whole entity graph.
  Use for brand identity, overall positioning, dominant themes.
- LOCAL: context is BFS-expanded posts and entities most similar to the query.
  Use for specific posts, hashtags, engagement metrics.
- COMPARE: data for multiple accounts. Contrast them — strengths, weaknesses,
  overlaps, differences.

Rules:
- Answer using the graph data — cite entity names, post IDs, hashtags
- Be concise but insightful — 2-4 paragraphs
- End with 1-2 actionable recommendations
- Return ONLY valid JSON, no markdown fences

JSON shape:
{
  "answer": "full plain-text answer",
  "highlighted_post_ids": ["post_id_1", "post_id_2"]
}"""

def ask_groq(query: str, graph_context: dict) -> dict:
    if not GROQ_CLIENT:
        return {"answer": "GROQ_API_KEY not set in .env",
                "highlighted_post_ids": [], "key_insight": ""}

    mode = graph_context.get("retrieval_mode", "local")

    if mode == "compare":
        slim = {
            "query": query,
            "retrieval_mode": "compare",
            "accounts": {
                u: {
                    "total_posts":   ctx.get("engagement", {}).get("total_posts", 0),
                    "avg_likes":     ctx.get("engagement", {}).get("avg_likes", 0),
                    "top_entities":  ctx.get("entities", {}).get("top_entities", [])[:8],
                    "top_hashtags":  ctx.get("hashtags", {}).get("top_by_frequency", [])[:6],
                    "content_types": ctx.get("entities", {}).get("content_types", {}),
                    "sentiment_mix": ctx.get("entities", {}).get("sentiment_mix", {}),
                    "communities": [
                        {"summary": c.get("summary", ""), "size": c["size"],
                         "top_entities": [m["name"] for m in c.get("members", [])[:5]]}
                        for c in ctx.get("communities", [])[:3]
                    ],
                }
                for u, ctx in graph_context.get("accounts", {}).items()
            },
            "shared_hashtags": graph_context.get("shared_hashtags", []),
            "shared_entities": graph_context.get("shared_entities", [])[:10],
        }
    elif mode == "global":
        slim = {
            "query":          query,
            "username":       graph_context.get("username"),
            "retrieval_mode": "global",
            "summary_stats":  graph_context.get("summary_stats", {}),
            "communities": [
                {"community_id": c["community_id"], "size": c["size"],
                 "avg_likes": c.get("avg_likes", 0), "summary": c.get("summary", ""),
                 "top_entities": [m["name"] for m in c.get("members", [])[:8]]}
                for c in graph_context.get("communities", [])[:6]
            ],
        }
    else:
        slim = {
            "query":          query,
            "username":       graph_context.get("username"),
            "retrieval_mode": "local",
            "summary_stats":  graph_context.get("summary_stats", {}),
            "seed_nodes":     graph_context.get("seed_nodes", []),
            "posts":          graph_context.get("subgraph", {}).get("posts", [])[:8],
            "entities":       graph_context.get("subgraph", {}).get("entities", [])[:15],
            "relationships":  graph_context.get("subgraph", {}).get("relationships", [])[:25],
        }

    try:
        resp = GROQ_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": (
                    "Question: " + query + "\n\nGraph RAG context:\n"
                    + json.dumps(slim, indent=2, default=str)
                )},
            ],
            temperature=0.3,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"answer": f"Groq error: {e}", "highlighted_post_ids": [], "key_insight": ""}

_LOAD_RE = re.compile(
    r"^\s*(?:load|open|use|switch\s+to|analyze|analyse|show)\s+@?([a-zA-Z0-9_.]+)\s*$",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"compare\s+@?([a-zA-Z0-9_.]+)\s+(?:and|vs\.?|with)\s+@?([a-zA-Z0-9_.]+)",
    re.IGNORECASE,
)

_MENTION_RE = re.compile(r"@([a-zA-Z0-9_.]+)")

_STOP_WORDS = {
    "what", "which", "who", "how", "when", "where", "why", "is", "are", "was",
    "the", "this", "that", "these", "those", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "overall", "brand", "account", "page", "profile", "posts", "post", "feed",
    "content", "identity", "strategy", "tell", "me", "show", "give", "get",
    "does", "do", "did", "has", "have", "had", "its", "their", "my", "our",
    "most", "best", "top", "all", "any", "some", "more", "less", "many",
    "instagram", "social", "media", "compare", "comparison", "between", "vs",
    "load", "open", "use", "analyze", "analyse", "engagement", "likes",
    "comments", "followers", "hashtag", "hashtags", "mention", "mentions",
    "similar", "same", "different", "difference", "similar", "like", "also",
    "i", "you", "we", "they", "he", "she", "it", "across", "both", "each",
}

_SCRAPE_RE = re.compile(
    r"^\s*(?:scrape|fetch|pull|get|grab|download|collect)\s+@?([a-zA-Z0-9_.]+)"
    r"(?:\s+(?:instagram|ig|insta|data|posts?|account|profile|feed))?\s*$",
    re.IGNORECASE,
)

def detect_scrape(text: str) -> str | None:
    """Explicit scrape/fetch command: 'scrape puma' or 'fetch puma instagram data'"""
    m = _SCRAPE_RE.match(text.strip())
    if m:
        name = m.group(1).lower()
        return name if name not in _STOP_WORDS else None
    return None

def detect_load(text: str) -> str | None:
    """Explicit load command: 'load nike'"""
    m = _LOAD_RE.match(text.strip())
    return m.group(1).lower() if m else None

def detect_compare(text: str) -> tuple[str, str] | None:
    """Explicit or implicit two-account comparison."""

    m = _COMPARE_RE.search(text)
    if m:
        return m.group(1).lower(), m.group(2).lower()

    m2 = re.search(
        r"@?([a-zA-Z0-9_.]+)\s+(?:vs\.?|versus)\s+@?([a-zA-Z0-9_.]+)",
        text, re.IGNORECASE,
    )
    if m2:
        u1 = m2.group(1).lower()
        u2 = m2.group(2).lower()
        if u1 not in _STOP_WORDS and u2 not in _STOP_WORDS:
            return u1, u2

    return None

def extract_mentioned_accounts(text: str) -> list[str]:
    """
    Pull Instagram account names out of any free-form query.

    Handles:
      @nike                       → ["nike"]
      "nike brand's identity"     → ["nike"]
      "adidas content strategy"   → ["adidas"]
      "compare nike and adidas"   → ["nike", "adidas"]   (via detect_compare)

    Returns deduplicated list, preserving order of first mention.
    """
    found: list[str] = []
    seen:  set[str]  = set()

    def _add(name: str):
        name = name.lower().strip("_.")
        if name and name not in _STOP_WORDS and name not in seen and len(name) >= 2:
            seen.add(name)
            found.append(name)

    for m in _MENTION_RE.finditer(text):
        _add(m.group(1))

    patterns = [
        r"([a-zA-Z0-9_.]{2,30})'?s?\s+(?:brand|account|page|profile|content|feed|posts?|identity|strategy|overall)",
        r"(?:about|for|of|is|analyze|analyse|load|open)\s+@?([a-zA-Z0-9_.]{2,30})",
        r"([a-zA-Z0-9_.]{2,30})\s+(?:brand|account|page|profile)",
        r"(?:scrape|fetch|pull|get|grab|download|collect)\s+@?([a-zA-Z0-9_.]{2,30})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            _add(m.group(1))

    return found

def _ensure_account(username: str) -> int:
    """Scrape + build graph if not cached. Returns post count."""
    cached = get_cached_accounts()
    if username not in cached:
        scrape_account(username, force=False)
    build_graph(username)
    pp = processed_path(username)
    return len(json.loads(pp.read_text())) if pp.exists() else 0

def _build_nodes_edges_single(username: str) -> tuple[list, list]:
    pp = processed_path(username)
    if not pp.exists():
        return [], []
    posts     = json.loads(pp.read_text())
    max_likes = max((p["like_count"] for p in posts), default=1)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen:  set[str]   = set()

    nodes.append({"id": username, "label": f"@{username}", "group": "brand", "size": 50})

    for p in posts:
        pid  = p["post_id"]
        size = int(14 + 34 * (p["like_count"] / max(max_likes, 1)))
        nodes.append({
            "id": pid, "label": f"Post\n{pid[:8]}", "group": "post", "size": size,
            "like_count": p["like_count"], "comment_count": p["comment_count"],
            "media_type": p["media_type"], "url": p["url"],
            "caption":    (p.get("caption") or "")[:180],
            "content_type": p.get("content_type", "other"),
            "sentiment":    p.get("sentiment", "neutral"),
        })
        edges.append({"from": username, "to": pid, "label": "POSTED"})

        for ent in p.get("graph_entities", []):
            eid = f"entity:{ent['id']}"
            if eid not in seen:
                nodes.append({
                    "id": eid, "label": ent.get("name", ent["id"])[:22],
                    "group": "entity", "entity_type": ent.get("type", "Topic"), "size": 20,
                })
                seen.add(eid)
            edges.append({"from": pid, "to": eid, "label": "CONTAINS_ENTITY"})

        for tag in p.get("hashtags", [])[:5]:
            tid = f"#{tag}"
            if tid not in seen:
                nodes.append({"id": tid, "label": tid[:18], "group": "hashtag", "size": 13})
                seen.add(tid)
            edges.append({"from": pid, "to": tid, "label": "HAS_HASHTAG"})

        month_id = f"month:{p['month']}"
        if month_id not in seen:
            nodes.append({"id": month_id, "label": p["month"], "group": "month", "size": 22})
            seen.add(month_id)
        edges.append({"from": pid, "to": month_id, "label": "BELONGS_TO"})

    return nodes, edges

def _build_nodes_edges_compare(usernames: list[str]) -> tuple[list, list]:
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    seen: set[str] = set()

    for i, username in enumerate(usernames):
        pp = processed_path(username)
        if not pp.exists():
            continue
        posts     = json.loads(pp.read_text())
        max_likes = max((p["like_count"] for p in posts), default=1)
        brand_id  = f"brand:{username}"

        if brand_id not in seen:
            all_nodes.append({
                "id": brand_id, "label": f"@{username}",
                "group": f"brand_{chr(ord('a')+i)}", "size": 50,
            })
            seen.add(brand_id)

        for p in posts:
            pid  = f"{username}:{p['post_id']}"
            size = int(14 + 34 * (p["like_count"] / max(max_likes, 1)))
            all_nodes.append({
                "id": pid, "label": f"{username[:5]}\n{p['post_id'][:6]}",
                "group": f"post_{chr(ord('a')+i)}", "size": size,
                "like_count": p["like_count"], "comment_count": p["comment_count"],
                "media_type": p["media_type"], "url": p["url"],
                "caption":    (p.get("caption") or "")[:180],
                "content_type": p.get("content_type", "other"),
                "sentiment":    p.get("sentiment", "neutral"),
            })
            all_edges.append({"from": brand_id, "to": pid, "label": "POSTED"})

            for ent in p.get("graph_entities", []):
                eid = f"entity:{ent['id']}"
                if eid not in seen:
                    all_nodes.append({
                        "id": eid, "label": ent.get("name", ent["id"])[:22],
                        "group": "entity", "entity_type": ent.get("type", "Topic"), "size": 20,
                    })
                    seen.add(eid)
                all_edges.append({"from": pid, "to": eid, "label": "CONTAINS_ENTITY"})

            for tag in p.get("hashtags", [])[:4]:
                tid = f"#{tag}"
                if tid not in seen:
                    all_nodes.append({"id": tid, "label": tid[:18], "group": "hashtag", "size": 13})
                    seen.add(tid)
                all_edges.append({"from": pid, "to": tid, "label": "HAS_HASHTAG"})

            month_id = f"month:{username}:{p['month']}"
            if month_id not in seen:
                all_nodes.append({"id": month_id, "label": p["month"], "group": "month", "size": 20})
                seen.add(month_id)
            all_edges.append({"from": pid, "to": month_id, "label": "BELONGS_TO"})

    return all_nodes, all_edges

def _render_graph_html(
    nodes: list,
    edges: list,
    highlight_ids: list[str] | None = None,
    styles: dict | None = None,
) -> str:
    if not nodes:
        return "<p style='color:#6b7280;padding:20px;font-family:system-ui'>No data.</p>"

    gs  = styles or GROUP_STYLES
    nj  = json.dumps(nodes)
    ej  = json.dumps(edges)
    gsj = json.dumps(gs)
    ecj = json.dumps(EDGE_COLORS)
    ewj = json.dumps(EDGE_WIDTHS)
    hj  = json.dumps(highlight_ids or [])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  *{{box-sizing:border-box}}
  html,body{{margin:0;padding:0;background:#060a0f;height:100%;overflow:hidden;
    font-family:system-ui,-apple-system,sans-serif}}

    background:rgba(6,10,15,.9);border:1px solid rgba(255,255,255,.1);
    border-radius:20px;padding:5px 16px;color:#6b7280;font-size:11px;
    backdrop-filter:blur(14px);white-space:nowrap;z-index:99;
    display:flex;gap:12px;align-items:center}}
  .hbadge{{color:#E040FB;font-weight:700;font-size:10px;
    border:1px solid #E040FB55;border-radius:4px;padding:1px 6px;letter-spacing:1px}}
  .hval{{color:#e6edf3;font-weight:600}}

    display:flex;gap:8px;z-index:99}}
  .cbtn{{background:rgba(6,10,15,.9);border:1px solid rgba(255,255,255,.12);
    border-radius:20px;padding:5px 14px;color:#6b7280;font-size:11px;cursor:pointer;
    backdrop-filter:blur(14px);transition:all .15s}}
  .cbtn:hover{{color:#e6edf3;border-color:rgba(255,255,255,.3)}}
  .cbtn.on{{color:#E040FB;border-color:#E040FB60}}

    border:1px solid rgba(255,255,255,.12);border-radius:12px;
    padding:13px 16px;color:#c9d1d9;font-size:11.5px;
    pointer-events:auto;display:none;z-index:300;max-width:320px;
    line-height:1.75;box-shadow:0 8px 40px rgba(0,0,0,.85)}}
  .tt{{font-weight:700;font-size:13px;color:#e6edf3;margin-bottom:6px}}
  .tr{{display:flex;justify-content:space-between;gap:14px;margin:2px 0}}
  .tl{{color:#6b7280}}.tv{{color:#58a6ff;font-weight:600}}
  .te{{color:#E040FB;font-size:9px;font-weight:700;text-transform:uppercase;
    letter-spacing:1.2px;margin-bottom:4px}}
  .tc{{margin-top:9px;padding-top:9px;border-top:1px solid rgba(255,255,255,.07);
    color:#6b7280;font-size:10.5px;font-style:italic;line-height:1.5}}
  .tlink{{display:inline-block;margin-top:10px;padding:5px 13px;
    background:rgba(255,69,0,.12);border:1px solid rgba(255,69,0,.35);
    border-radius:8px;color:#FF4500;text-decoration:none;font-weight:600;
    font-size:11px;transition:background .15s}}
  .tlink:hover{{background:rgba(255,69,0,.28)}}
</style>
</head>
<body>
<div id="hud">
  <span class="hbadge">⬡ GRAPH RAG</span>
  <span><span class="hval" id="nc">–</span> nodes</span>
  <span><span class="hval" id="ec">–</span> edges</span>
  <span id="hls">&nbsp;·&nbsp;<span class="hval" id="hlc">0</span> highlighted</span>
</div>
<div id="ctrl">
  <button class="cbtn on" id="physBtn" onclick="togglePhysics()">⚙ Physics ON</button>
  <button class="cbtn" onclick="net.fit({{animation:{{duration:600,easingFunction:'easeInOutQuad'}}}})">⊙ Fit</button>
  <button class="cbtn" onclick="clearHL()">✕ Clear</button>
</div>
<div id="net"></div>
<div id="tip"></div>
<script>
const GS={gsj},EC={ecj},EW={ewj};
const SHAPES={{
  brand:'star',brand_a:'star',brand_b:'star',brand_c:'star',brand_d:'star',
  post:'dot',post_a:'dot',post_b:'dot',post_c:'dot',post_d:'dot',
  entity:'hexagon',hashtag:'diamond',mention:'triangle',
  audience:'triangleDown',month:'square',
  media_type:'ellipse',content_type:'ellipse',sentiment:'ellipse'
}};
const vN={nj}.map(n=>{{
  const s=GS[n.group]||{{color:n._brand_color||'#888',glow:n._brand_color||'#888',size:12}};
  const col=n._brand_color||s.color;
  return{{
    id:n.id,label:n.label,group:n.group,
    size:n.size||s.size||12,_base:n.size||s.size||12,
    color:{{background:col,border:col,
      highlight:{{background:col,border:'#fff'}},
      hover:{{background:col,border:'#fff'}}}},
    shape:SHAPES[n.group]||'dot',
    font:{{color:'#e6edf3',
      size:(n.group==='brand'||n.group.startsWith('brand_'))?17:n.group==='entity'?11:9,
      bold:(n.group==='brand'||n.group.startsWith('brand_')||n.group==='entity'),
      strokeWidth:3,strokeColor:'rgba(6,10,15,.85)'}},
    borderWidth:(n.group==='brand'||n.group.startsWith('brand_'))?4:2,
    shadow:{{enabled:true,color:s.glow||col,
      size:(n.group==='brand'||n.group.startsWith('brand_'))?32:n.group==='entity'?16:9,x:0,y:0}},
    _lk:n.like_count,_cm:n.comment_count,_mt:n.media_type,
    _et:n.entity_type,_ct:n.content_type,_sn:n.sentiment,
    _url:n.url,_cap:n.caption,
  }};
}});
const vE={ej}.map(e=>({{
  from:e.from,to:e.to,
  color:{{color:EC[e.label]||'#ffffff15',highlight:'#ffffff70',hover:'#ffffff50',inherit:false}},
  width:EW[e.label]||1,
  arrows:{{to:{{enabled:true,scaleFactor:.45,type:'arrow'}}}},
  smooth:{{enabled:true,type:'continuous',roundness:.38}},
  shadow:{{enabled:true,color:'rgba(0,0,0,.3)',size:4,x:0,y:0}},
  title:e.label,
}}));
const ds={{nodes:new vis.DataSet(vN),edges:new vis.DataSet(vE)}};
const net=new vis.Network(document.getElementById('net'),ds,{{
  physics:{{enabled:true,
    forceAtlas2Based:{{gravitationalConstant:-80,centralGravity:.008,
      springLength:160,springConstant:.055,damping:.42,avoidOverlap:1}},
    maxVelocity:60,minVelocity:.08,solver:'forceAtlas2Based',
    stabilization:{{enabled:true,iterations:1400,updateInterval:20,fit:true}}}},
  interaction:{{hover:true,tooltipDelay:80,zoomView:true,dragView:true,hideEdgesOnDrag:true}},
  nodes:{{scaling:{{min:8,max:52}}}},edges:{{scaling:{{min:1,max:4}}}},
}});
document.getElementById('nc').textContent=vN.length;
document.getElementById('ec').textContent=vE.length;
let physOn=true;
net.on('stabilizationIterationsDone',()=>{{ net.setOptions({{physics:{{maxVelocity:6,minVelocity:.04}}}}); }});
function togglePhysics(){{
  physOn=!physOn;
  net.setOptions({{physics:{{enabled:physOn,maxVelocity:physOn?6:0}}}});
  const b=document.getElementById('physBtn');
  b.textContent=physOn?'⚙ Physics ON':'⚙ Physics OFF';
  b.classList.toggle('on',physOn);
}}
const tip=document.getElementById('tip');
let pinned=false,htimer=null;
tip.addEventListener('mouseenter',()=>{{pinned=true;if(htimer)clearTimeout(htimer);}});
tip.addEventListener('mouseleave',()=>{{pinned=false;tip.style.display='none';}});
net.on('hoverNode',p=>{{
  const n=ds.nodes.get(p.node);
  if(!n)return;
  let h='<div class="tt">'+n.label.replace(/\\n/g,' ')+'</div>';
  if(n._et) h+='<div class="te">'+n._et+'</div>';
  h+='<div class="tr"><span class="tl">Type</span><span class="tv">'+n.group+'</span></div>';
  if(n._lk!=null) h+='<div class="tr"><span class="tl">Likes</span><span class="tv">'+n._lk.toLocaleString()+'</span></div>';
  if(n._cm!=null) h+='<div class="tr"><span class="tl">Comments</span><span class="tv">'+n._cm.toLocaleString()+'</span></div>';
  if(n._mt) h+='<div class="tr"><span class="tl">Media</span><span class="tv">'+n._mt+'</span></div>';
  if(n._ct) h+='<div class="tr"><span class="tl">Content type</span><span class="tv">'+n._ct+'</span></div>';
  if(n._sn) h+='<div class="tr"><span class="tl">Sentiment</span><span class="tv">'+n._sn+'</span></div>';
  if(n._cap) h+='<div class="tc">&ldquo;'+n._cap.substring(0,150)+'&hellip;&rdquo;</div>';
  if(n._url) h+='<a class="tlink" href="'+n._url+'" target="_blank" rel="noopener">↗ Open on Instagram</a>';
  tip.innerHTML=h;tip.style.display='block';
  const pos=net.canvasToDOM(net.getPosition(p.node));
  const tw=320,mg=16;
  let lx=pos.x+mg;if(lx+tw>window.innerWidth-10)lx=pos.x-tw-mg;
  let ty=Math.max(50,pos.y-10);
  if(ty+tip.offsetHeight>window.innerHeight-10)ty=window.innerHeight-tip.offsetHeight-10;
  tip.style.left=lx+'px';tip.style.top=ty+'px';
}});
net.on('blurNode',()=>{{ if(pinned)return; htimer=setTimeout(()=>{{if(!pinned)tip.style.display='none';}},260); }});
net.on('click',p=>{{ if(!p.nodes.length)return; net.focus(p.nodes[0],{{scale:1.5,animation:{{duration:450,easingFunction:'easeInOutQuad'}}}}); }});
let ptimer=null;
function hlNodes(ids){{
  const s=new Set(ids);
  if(!s.size){{clearHL();return;}}
  document.getElementById('hls').style.display='';
  document.getElementById('hlc').textContent=s.size;
  ds.nodes.update(ds.nodes.map(n=>{{ const hi=s.has(n.id); return{{id:n.id,opacity:hi?1:.07,size:hi?(n._base||12)*1.7:(n._base||12)}}; }}));
  let grow=true,step=0;
  if(ptimer)clearInterval(ptimer);
  ptimer=setInterval(()=>{{
    step+=grow?1:-1;
    if(step>=7)grow=false;if(step<=0)grow=true;
    ds.nodes.update([...s].map(id=>{{ const n=ds.nodes.get(id); return{{id,size:(n._base||12)*1.7+step*1.8}}; }}));
  }},75);
  setTimeout(()=>net.fit({{nodes:[...s],animation:{{duration:650,easingFunction:'easeInOutQuad'}}}}),130);
}}
function clearHL(){{
  if(ptimer){{clearInterval(ptimer);ptimer=null;}}
  document.getElementById('hls').style.display='none';
  ds.nodes.update(ds.nodes.map(n=>({{id:n.id,opacity:1,size:n._base||n.size||12}})));
  net.fit({{animation:{{duration:500,easingFunction:'easeInOutQuad'}}}});
}}
const _init={hj};
if(_init.length){{ net.once('stabilizationIterationsDone',()=>setTimeout(()=>hlNodes(_init),350)); }}
window.addEventListener('message',e=>{{
  if(!e.data||e.data.type!=='graphrag_highlight')return;
  if(e.data.ids&&e.data.ids.length)hlNodes(e.data.ids);else clearHL();
}});
</script>
</body>
</html>"""

def render_graph_html(username: str, highlight_ids: list[str] | None = None) -> str:
    nodes, edges = _build_nodes_edges_single(username)
    return _render_graph_html(nodes, edges, highlight_ids, GROUP_STYLES)

def render_comparison_graph_html(usernames: list[str]) -> str:
    nodes, edges = _build_nodes_edges_compare(usernames)
    return _render_graph_html(nodes, edges, None, {**GROUP_STYLES, **COMPARISON_BRAND_STYLES})

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#060a0f; }
  [data-testid="stHeader"]           { display:none; }
  section[data-testid="stSidebar"]   { display:none; }
  .block-container { padding:0 !important; max-width:100% !important; }
  div[data-testid="column"] { padding:0 !important; }
  .topbar {
    display:flex;align-items:center;gap:10px;padding:12px 18px;
    background:rgba(6,10,15,.97);border-bottom:1px solid rgba(255,255,255,.07);
  }
  .topbar-title { font-size:14px;font-weight:700;color:#e6edf3;font-family:system-ui }
  .topbar-badge {
    font-size:9px;font-weight:700;color:#E040FB;
    border:1px solid #E040FB55;border-radius:4px;padding:2px 6px;letter-spacing:1px
  }
  .topbar-acct { margin-left:auto;font-size:11px;color:#6b7280;font-family:system-ui }
  .topbar-acct b { color:#58a6ff }
  .msg-user {
    background:rgba(30,144,255,.09);border:1px solid rgba(30,144,255,.18);
    border-radius:16px 16px 4px 16px;padding:10px 14px;margin:5px 0 5px 32px;
    color:#e6edf3;font-size:13px;font-family:system-ui;line-height:1.65;
  }
  .msg-bot {
    background:rgba(224,64,251,.07);border:1px solid rgba(224,64,251,.15);
    border-radius:16px 16px 16px 4px;padding:10px 14px;margin:5px 32px 5px 0;
    color:#e6edf3;font-size:13px;font-family:system-ui;line-height:1.65;
  }
  .msg-sys {
    text-align:center;color:#4b5563;font-size:11px;
    font-family:system-ui;padding:5px 0;font-style:italic;
  }
  .stTextInput>div>div>input {
    background:rgba(22,27,34,.95) !important;
    border:1px solid rgba(255,255,255,.1) !important;
    color:#e6edf3 !important;border-radius:12px !important;
    font-size:13px !important;padding:10px 14px !important;
  }
  .stTextInput>div>div>input:focus {
    border-color:rgba(224,64,251,.45) !important;
    box-shadow:0 0 0 3px rgba(224,64,251,.07) !important;
  }
  .stButton>button {
    background:rgba(224,64,251,.13) !important;
    border:1px solid rgba(224,64,251,.38) !important;
    color:#E040FB !important;border-radius:12px !important;
    font-size:18px !important;font-weight:700 !important;
    width:100% !important;height:44px !important;
    transition:background .15s !important;
  }
  .stButton>button:hover { background:rgba(224,64,251,.25) !important; }
</style>
""", unsafe_allow_html=True)

for k, v in {
    "messages":         [],
    "active_account":   None,
    "compare_accounts": [],
    "highlight_ids":    [],
    "graph_html":       "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

left, right = st.columns([4, 6], gap="small")

with left:
    acct      = st.session_state.active_account
    cmp_accts = st.session_state.compare_accounts

    if cmp_accts:
        acct_html = " <span style='color:#6b7280'>vs</span> ".join(
            f"<b>@{u}</b>" for u in cmp_accts
        )
    elif acct:
        acct_html = f"<b>@{acct}</b>"
    else:
        acct_html = ""

    st.markdown(
        '<div class="topbar">'
        '<span style="font-size:17px">⬡</span>'
        '<span class="topbar-title">Graph RAG</span>'
        '<span class="topbar-badge">RAG</span>'
        f'<span class="topbar-acct">{acct_html}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    chat = st.container(height=660, border=False)
    with chat:
        for msg in st.session_state.messages:
            role    = msg["role"]
            content = msg["content"]
            if role == "system":
                st.markdown(f'<div class="msg-sys">{content}</div>', unsafe_allow_html=True)
            elif role == "user":
                st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-bot">{content}</div>', unsafe_allow_html=True)

    col_inp, col_send = st.columns([6, 1])
    with col_inp:
        prompt = st.text_input(
            "msg", key="prompt_input",
            placeholder="Ask anything…",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("↑", use_container_width=True)

    final_prompt = prompt.strip() if (send and prompt.strip()) else None

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})

    load_target    = detect_load(final_prompt)
    scrape_target  = detect_scrape(final_prompt)
    compare_target = detect_compare(final_prompt)

    if scrape_target:
        username = scrape_target
        with st.spinner(f"Scraping @{username} from Instagram — this may take 1–2 min…"):
            post_count = _ensure_account(username)

        st.session_state.active_account   = username
        st.session_state.compare_accounts = []
        st.session_state.highlight_ids    = []
        st.session_state.graph_html       = render_graph_html(username)

        comms = load_communities(username)
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"Scraped and loaded **@{username}** — {post_count} posts"
                + (f", {len(comms)} communities" if comms else "")
                + ". Graph is live on the right. You can now ask questions about this account."
            ),
        })

    elif load_target:
        username = load_target
        with st.spinner(f"Loading @{username}…"):
            post_count = _ensure_account(username)

        st.session_state.active_account   = username
        st.session_state.compare_accounts = []
        st.session_state.highlight_ids    = []
        st.session_state.graph_html       = render_graph_html(username)

        comms = load_communities(username)
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"Loaded **@{username}** — {post_count} posts"
                + (f", {len(comms)} communities" if comms else "")
                + ". Graph is live on the right."
            ),
        })

    elif compare_target:
        u1, u2 = compare_target
        with st.spinner(f"Loading @{u1} and @{u2}…"):
            c1 = _ensure_account(u1)
            c2 = _ensure_account(u2)

        st.session_state.compare_accounts = [u1, u2]
        st.session_state.active_account   = None
        st.session_state.highlight_ids    = []
        st.session_state.graph_html       = render_comparison_graph_html([u1, u2])

        with st.spinner("Comparing accounts…"):
            graph_ctx = get_comparison_context([u1, u2])
            graph_ctx["retrieval_mode"] = "compare"
        with st.spinner("Generating answer…"):
            result = ask_groq(final_prompt, graph_ctx)

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"Loaded **@{u1}** ({c1} posts) and **@{u2}** ({c2} posts). "
                "Both graphs shown on the right — purple hexagons are shared entities.\n\n"
                + result.get("answer", "")
            ),
        })

    else:
        mentioned = extract_mentioned_accounts(final_prompt)
        current_accounts = (
            st.session_state.compare_accounts or
            ([st.session_state.active_account] if st.session_state.active_account else [])
        )

        new_account = None
        for name in mentioned:
            if name not in current_accounts:
                new_account = name
                break

        if new_account:
            with st.spinner(f"Auto-loading @{new_account}…"):
                post_count = _ensure_account(new_account)
            st.session_state.active_account   = new_account
            st.session_state.compare_accounts = []
            st.session_state.graph_html       = render_graph_html(new_account)
            st.session_state.messages.append({
                "role": "system",
                "content": f"Auto-loaded @{new_account} ({post_count} posts)",
            })

        if st.session_state.compare_accounts:

            with st.spinner("Querying both accounts…"):
                graph_ctx = get_comparison_context(st.session_state.compare_accounts)
                graph_ctx["retrieval_mode"] = "compare"
            with st.spinner("Generating answer…"):
                result = ask_groq(final_prompt, graph_ctx)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.get("answer", "No answer returned."),
            })

        elif st.session_state.active_account:
            username = st.session_state.active_account
            with st.spinner("Traversing knowledge graph…"):
                graph_ctx = query_graph(
                    username, final_prompt,
                    top_k_seeds=5, top_k_nodes=30, include_posts=8,
                )
            with st.spinner("Generating answer…"):
                result = ask_groq(final_prompt, graph_ctx)

            answer = result.get("answer", "No answer returned.")
            h_ids  = list(set(
                result.get("highlighted_post_ids", []) +
                graph_ctx.get("highlight_ids", [])
            ))[:10]

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.highlight_ids = h_ids
            st.session_state.graph_html    = render_graph_html(username, highlight_ids=h_ids)

        else:

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "I couldn't figure out which Instagram account you're asking about. "
                    "Try typing an account name, e.g. **what is puma's brand identity?** "
                    "or **scrape puma** to load fresh data."
                ),
            })

    st.rerun()

with right:
    if st.session_state.graph_html:
        components.html(st.session_state.graph_html, height=920, scrolling=False)
    else:
        st.markdown("""
        <div style="height:920px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:16px;background:#0d1117;border-radius:12px;
                    border:1px solid rgba(255,255,255,.05)">
          <div style="font-size:56px;opacity:.1">⬡</div>
        </div>
        """, unsafe_allow_html=True)