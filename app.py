import os
import re
import json
import logging
from functools import lru_cache
from collections import Counter

import numpy as np
from flask import Flask, request, jsonify, render_template
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
BUCKET_NAME    = os.getenv("BUCKET_NAME", "legal-pdfs")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# APP / CLIENTS 
app = Flask(__name__)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

# helpers 
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")  # tokens >= 3 letters

def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

def _unit(v):
    """L2-normalize a vector for stable cosine."""
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / (n + 1e-12)).tolist()

def fuzzy_score(a: str, b: str) -> float:
    """Safer than partial_ratio for full queries."""
    return fuzz.token_set_ratio(a, b) / 100.0

def get_embedding(text, model="text-embedding-ada-002"):
    emb = oa.embeddings.create(model=model, input=text).data[0].embedding
    return _unit(emb)



# home
@app.route("/")
def home():
    return render_template("index.html")

#  Suggestion engine 

# Sentence/paragraph chopping for snippets
SENT_MIN, SENT_MAX = 25, 400
DOC_TOP_N = 10
_sentence_split_re = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“"(\[])')

def split_sentences(text: str):
    if not text:
        return []
    t = re.sub(r'\s+', ' ', text).strip()
    parts = _sentence_split_re.split(t)
    return [s.strip() for s in parts if SENT_MIN <= len(s.strip()) <= SENT_MAX]

PAR_MIN, PAR_MAX = 60, 800
_par_split_re = re.compile(r'(?:\r?\n\s*\r?\n)+')

def split_paragraphs(text: str):
    if not text:
        return []
    parts = _par_split_re.split(text)
    parts = [re.sub(r'\s+', ' ', p).strip() for p in parts]
    return [p for p in parts if PAR_MIN <= len(p) <= PAR_MAX]

@lru_cache(maxsize=1)
def load_docs_cache():
    """Load docs once, normalize embeddings for stable cosine."""
    rows = sb.table("legal_documents").select("id,title,content,file_url,embedding").execute().data or []
    for d in rows:
        emb = d.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = None
        d["embedding"] = _unit(emb) if emb is not None else None
        d["title"] = d.get("title") or ""
        d["content"] = d.get("content") or ""
    return rows

@lru_cache(maxsize=1)
def build_corpus_stats():
    """Corpus-driven vocab & dynamic stopwords; no hardcoded domain terms."""
    docs = load_docs_cache()
    df = Counter()
    for d in docs:
        toks = set(tokenize(d["title"] + " " + d["content"]))
        for t in toks:
            df[t] += 1
    stop = {t for t, _ in df.most_common(120)}  # absorb very common legal fillers
    vocab = {t for t, c in df.items() if c >= 1 and t not in stop}
    return {"vocab": vocab, "stop": stop, "df": df}

# --- light morphology + typo/prefix snap to vocab ---
#safe singularization
def _morph_to_vocab(tok: str, vocab: set):
    if tok.endswith("ies") and len(tok) > 4 and (tok[:-3] + "y") in vocab:
        return tok[:-3] + "y"
    if tok.endswith("ses") and len(tok) > 4 and tok[:-2] in vocab:
        return tok[:-2]
    if tok.endswith("es") and len(tok) > 3 and tok[:-2] in vocab:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3 and tok[:-1] in vocab:
        return tok[:-1]
    return tok

#length aware typo check
#when gives a user tok decide whether vtok is acceptable
def _proj_accept(tok: str, vtok: str):
    L = max(len(tok), len(vtok))
    sim = distance.Levenshtein.normalized_similarity(tok, vtok)
    if L >= 4 and (tok[0] != vtok[0] or tok[-1] != vtok[-1]):
        return False, sim
    if L <= 4:
        ok = sim >= 0.75
    elif L <= 6:
        ok = sim >= 0.80
    elif L <= 10:
        ok = sim >= 0.86
    else:
        ok = sim >= 0.90
    return ok, sim

#if token length ≥ 4 and first/last chars don’t match, reject early.
def project_query_to_corpus(raw_query: str) -> str:
    """Map tokens to close forms that actually exist in your corpus."""
    st = build_corpus_stats()
    vocab, df = st["vocab"], st["df"]
    q_tokens = tokenize(raw_query)
    if not vocab or not q_tokens:
        return raw_query

    projected = []
    for tok in q_tokens:
        mtok = _morph_to_vocab(tok, vocab)
        if mtok in vocab:
            projected.append(mtok)
            continue

        best, best_sim = None, 0.0
        for vtok in vocab:
            if abs(len(vtok) - len(tok)) > 2:
                continue
            ok, sim = _proj_accept(tok, vtok)
            if ok and sim > best_sim:
                best, best_sim = vtok, sim

        if not best and len(tok) >= 4:
            candidates = [v for v in vocab if v.startswith(tok) and (len(v) - len(tok)) <= 4 and df[v] >= 1]
            if candidates:
                best = sorted(candidates, key=lambda x: (len(x), -df[x]))[0]

        projected.append(best if best else tok)

    return " ".join(projected)
#measures how many content tokens from query are found in vocab
def corpus_coverage_ratio(query: str) -> float:
    st = build_corpus_stats()
    q_tokens = [t for t in tokenize(query) if t not in st["stop"]]
    if not q_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in st["vocab"])
    return hits / len(q_tokens)

def doc_hybrid_score(q, q_emb, d):
    """Doc-level score for preselecting docs."""
    title = (d.get("title") or "").lower()
    content = (d.get("content") or "").lower()
    emb = d.get("embedding")

    cos = float(cosine_similarity([q_emb], [emb])[0][0]) if (q_emb is not None and emb is not None) else 0.0
    f_title = fuzzy_score(q.lower(), title)
    f_cont  = fuzzy_score(q.lower(), content[:1500])
    return 0.7 * cos + 0.15 * f_title + 0.15 * f_cont, cos

# --- Leading stopword gate ---
LEADING_STOPWORDS = {
    "the","a","an","of","and","or","if","to","in","on","for","with","by",
    "is","are","be","was","were","at","from","as","that","this","these","those","it","about"
}
_WORDS_ANY = re.compile(r"[a-z]+")

def _tokens_all(s: str):
    return _WORDS_ANY.findall((s or "").lower())

def _strip_leading_stopwords(tokens):
    i = 0
    while i < len(tokens) and tokens[i] in LEADING_STOPWORDS:
        i += 1
    return tokens[i:], i  # (tail, dropped)

# --- "X vs Y" title booster ---
VS_RE = re.compile(r"\b([a-z][a-z\-]{2,})\s+v(?:s\.?)?\s+([a-z][a-z\-]{1,})\b")

#title
def _title_hits_for_vs(raw_q: str, docs, top_n=1):
    m = VS_RE.search((raw_q or '').lower())
    if not m:
        return []
    a, b = m.group(1), m.group(2)

    hits = []
    for d in docs:
        title = (d.get("title") or "").lower()
        sa = max(fuzz.partial_ratio(a, title), fuzz.token_set_ratio(a, title)) / 100.0
        sb = max(fuzz.partial_ratio(b, title), fuzz.token_set_ratio(b, title)) / 100.0
        if sa >= 0.80 and sb >= 0.80:
            score = 0.5 * sa + 0.5 * sb
            hits.append((score, d))

    hits.sort(key=lambda x: x[0], reverse=True)
    suggestions = []
    for score, d in hits[:top_n]:
        text = (d.get("content") or "").strip()
        chunk = split_paragraphs(text)
        if not chunk:
            ss = split_sentences(text)
            chunk = ss[:1] if ss else []
        snippet = (chunk[0] if chunk else "")[:400]
        suggestions.append({
            "text": snippet,
            "title": d.get("title") or "",
            "url": d.get("file_url") or "",
            "score": score + 0.3
        })
    return suggestions

#  /suggest 
@app.route("/suggest", methods=["GET"])
def suggest():
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or 8)

    if len(q) < 2:
        return jsonify({"q": q, "suggestions": []})

    # If only leading stopwords so far, wait
    toks_all = _tokens_all(q)
    tail_tokens, dropped = _strip_leading_stopwords(toks_all)
    if dropped > 0 and not tail_tokens:
        return jsonify({"q": q, "suggestions": []})

    q_base = " ".join(tail_tokens) if tail_tokens else q
    docs = load_docs_cache()

    # Title booster for "X vs Y" queries
    vs_suggestions = _title_hits_for_vs(q_base, docs, top_n=1)

    # Project into corpus space
    q_proj = project_query_to_corpus(q_base)

    # Focus on content terms (drop dynamic stopwords)
    st = build_corpus_stats()
    content_terms = [t for t in tokenize(q_proj) if t not in st["stop"]]
    key_q = " ".join(content_terms) if content_terms else q_proj.lower()

    # original-content key (keeps plurals like "bulls/bullocks")
    orig_content_terms = [t for t in tokenize(q_base) if t not in st["stop"]]
    orig_key_q = " ".join(orig_content_terms) if orig_content_terms else q_base.lower()

    # Coverage check; BUT do not block if we already have a strong VS title hit
    coverage = corpus_coverage_ratio(q_proj) if len(q_proj) > 3 else 1.0
    out_of_corpus = (coverage == 0.0 and not vs_suggestions)

    # Dynamic thresholds: block OOD single tokens (e.g., "batman")
    if out_of_corpus and len(content_terms) <= 2:
        logging.info("Suggest blocked by OOD guard: q=%r key_q=%r coverage=%.2f", q, key_q, coverage)
        return jsonify({"q": q, "suggestions": vs_suggestions[:k]})

    if out_of_corpus and len(content_terms) <= 3:
        fuzzy_cut = 0.72
        doc_cos_cut = 0.18
    else:
        fuzzy_cut = 0.80 if out_of_corpus else 0.70
        doc_cos_cut = 0.30 if out_of_corpus else 0.15

    # Embed: choose the richer of projected vs original-content keys
    q_eff = key_q if len(key_q) >= len(orig_key_q) else orig_key_q
    q_emb = None
    if len(q_eff) >= 4:
        try:
            q_emb = get_embedding(q_eff)
        except Exception:
            q_emb = None

    # Preselect candidate docs
    scored = []
    for d in docs:
        total, cos = doc_hybrid_score(q_eff, q_emb, d)
        scored.append((total, cos, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    # One best snippet per document
    best_by_doc = {}
    for total, d_cos, d in scored[:DOC_TOP_N]:
        text = d.get("content") or ""
        chunks = split_paragraphs(text)
        if len(chunks) < 2:
            chunks = split_sentences(text)

        best_chunk = None
        best_score = -1.0

        for chunk in chunks:
            chunk_lc = chunk.lower()
            # use the best fuzzy score across projected and original-content keys
            f = max(
                fuzzy_score(key_q,      chunk_lc),
                fuzzy_score(orig_key_q, chunk_lc)
            )
            if f < fuzzy_cut:
                continue
            if q_emb is not None and d_cos < doc_cos_cut:
                continue

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

    body_suggestions = sorted(best_by_doc.values(), key=lambda s: s["score"], reverse=True)[:k]

    # Merge VS-title hits (if any) at the top, de-dup by URL
    seen = set()
    merged = []
    for s in vs_suggestions + body_suggestions:
        url = s.get("url")
        if url in seen:
            continue
        seen.add(url)
        merged.append(s)
        if len(merged) >= k:
            break

    logging.info("Suggest q=%r base=%r proj=%r key=%r coverage=%.2f OOD=%s n=%d",
                 q, q_base, q_proj, key_q, coverage, out_of_corpus, len(merged))
    return jsonify({"q": q, "suggestions": merged})

# ======= MAIN =======
if __name__ == "__main__":
    print("Supabase URL:", SUPABASE_URL)
    print("OpenAI key starts with:", (OPENAI_API_KEY or "")[:8], "…")
    try:
        load_docs_cache()
        build_corpus_stats()
    except Exception as e:
        logging.warning("Warmup failed: %s", e)
    app.run(debug=True)
