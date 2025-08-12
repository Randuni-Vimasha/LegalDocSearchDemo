import json
from flask import Flask, request, jsonify, render_template
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from dotenv import load_dotenv
load_dotenv()
import os
# ======= HARD-CODED KEYS =======

# ===========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
BUCKET_NAME    = os.getenv("BUCKET_NAME", "legal-pdfs")  #




app = Flask(__name__)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    return oa.embeddings.create(model=model, input=text).data[0].embedding

def hybrid_score(query, q_emb, doc):
    title   = (doc.get("title") or "").lower()
    content = (doc.get("content") or "").lower()
    url     = doc.get("file_url") or ""
    emb     = doc.get("embedding")

    # if vector come back as list, or JSON string 
    if isinstance(emb, str):
        emb = json.loads(emb)

    cos = float(cosine_similarity([q_emb], [emb])[0][0])
    f_title = fuzz.token_sort_ratio(query.lower(), title) / 100.0
    f_cont  = fuzz.partial_ratio(query.lower(), content[:1000]) / 100.0
    score = 0.5*cos + 0.2*f_title + 0.3*f_cont

    return {
        "title": doc.get("title"),
        "url": url,
        "score": score,
        "cosine": cos,
        "fuzzy_title": f_title,
        "fuzzy_content": f_cont,
        "snippet": content[:500]
    }

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/search", methods=["GET", "POST"])
def search():
    # accept GET (for realtime) and POST (for button submit)
    if request.method == "GET":
        query = (request.args.get("q") or "").strip()
        top_k = int(request.args.get("k") or 5)
        #q=query and k=top pdf results
    else:
        data = request.get_json(force=True)
        query = (data.get("q") or "").strip()
        top_k = int(data.get("k") or 5)

    # for realtime UX: if empty query, return empty list (not a 400)
    if not query:
        return jsonify({"query": "", "results": []})

    q_emb = get_embedding(query)
    docs = sb.table("legal_documents").select("*").execute().data or []

    results = [hybrid_score(query, q_emb, d) for d in docs]
    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"query": query, "results": results[:top_k]})

import re#split into sentences
from functools import lru_cache#cash memory

# --- helpers for suggestions (sentence level built from doc content) ---

SENT_MIN = 25      # ignore tiny fragments
SENT_MAX = 400     # ignore very long blobs
DOC_TOP_N = 10     # best docs to consider per query before sentence scan

_sentence_split_re = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“"(\[])')

def split_sentences(text: str):
    if not text:
        return []
    # normalize whitespace, preserve caps for a better split heuristic
    t = re.sub(r'\s+', ' ', text).strip()
    parts = _sentence_split_re.split(t)
    # length filter
    return [s.strip() for s in parts if SENT_MIN <= len(s.strip()) <= SENT_MAX]

#paragraphs for suggestions
PAR_MIN, PAR_MAX = 60, 800
_par_split_re = re.compile(r'(?:\r?\n\s*\r?\n)+')  # blank-line paragraph breaks

def split_paragraphs(text: str):
    if not text:
        return []
    parts = _par_split_re.split(text)
    parts = [re.sub(r'\s+', ' ', p).strip() for p in parts]
    return [p for p in parts if PAR_MIN <= len(p) <= PAR_MAX]


@lru_cache(maxsize=1)
def load_docs_cache():
    """Load all docs once, keep in memory."""
    docs = sb.table("legal_documents").select("title,content,file_url,embedding").execute().data or []
    # normalize embedding types
    for d in docs:
        emb = d.get("embedding")
        if isinstance(emb, str):
            try:
                d["embedding"] = json.loads(emb)
            except Exception:
                pass
    return docs

def doc_hybrid_score(q, q_emb, d):
    title = (d.get("title") or "").lower()
    content = (d.get("content") or "").lower()
    emb = d.get("embedding")

    # cosine (doc-level)
    cos = 0.0
    if q_emb is not None and emb is not None:
        cos = float(cosine_similarity([q_emb], [emb])[0][0])

    f_title = fuzz.token_sort_ratio(q.lower(), title) / 100.0
    f_cont  = fuzz.partial_ratio(q.lower(), content[:1500]) / 100.0
    return 0.6 * cos + 0.25 * f_title + 0.15 * f_cont, cos  # return both total and cosine

@app.route("/suggest", methods=["GET"])
def suggest():
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or 8)

    if len(q) < 2:
        return jsonify({"q": q, "suggestions": []})

    # embed only when it’s useful (short prefixes rely on fuzzy)
    q_emb = None
    if len(q) >= 4:
        try:
            q_emb = get_embedding(q)
        except Exception:
            q_emb = None

    docs = load_docs_cache()

    # Rank docs first (doc-level hybrid score)
    scored = []
    for d in docs:
        total, cos = doc_hybrid_score(q, q_emb, d)
        scored.append((total, cos, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep only ONE best snippet per document
    best_by_doc = {}
    q_lower = q.lower()

    for total, d_cos, d in scored[:DOC_TOP_N]:
        text = d.get("content") or ""
        chunks = split_paragraphs(text)
        if len(chunks) < 2:
            chunks = split_sentences(text)

        best_chunk = None
        best_score = -1.0

        for chunk in chunks:
            # quick fuzzy prefilter
            f = fuzz.partial_ratio(q_lower, chunk.lower()) / 100.0
            if f < 0.30:
                continue
            # combine doc cosine + local fuzzy
            s = (0.6 * d_cos + 0.4 * f) if q_emb is not None else f
            if s > best_score:
                best_score = s
                best_chunk = chunk

        if best_chunk:
            key = d.get("id") or d.get("title") or d.get("file_url")
            best_by_doc[key] = {
                "text": best_chunk,
                "title": d.get("title") or "",
                "url": d.get("file_url") or "",
                "score": best_score
            }

    # Rank the single-best snippets from each doc and return top-k
    suggestions = sorted(best_by_doc.values(), key=lambda s: s["score"], reverse=True)[:k]
    return jsonify({"q": q, "suggestions": suggestions})



if __name__ == "__main__":
    print("Supabase URL:", SUPABASE_URL)
    print("OpenAI key starts with:", OPENAI_API_KEY[:8], "…")
    app.run(debug=True)

