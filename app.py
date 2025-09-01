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
# NOTE: allow numbers and forms like 48(4) in tokens (fixes "section 32"/"section 839"/"48(4)")
TOKEN_RE = re.compile(r"(?:[a-z][a-z\-]{2,}|\d+(?:\([0-9a-z]{1,3}\))?)", re.IGNORECASE)
WORD_BOUNDARY_CACHE = {}

def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

def _unit(v):
    """L2-normalize a vector for stable cosine."""
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(arr)
    return (arr / (n + 1e-12)).tolist()

def _word_re(tok: str):
    r = WORD_BOUNDARY_CACHE.get(tok)
    if r is None:
        # Use custom non-word guards so tokens like "48(4)" match (plain \b fails at ')')
        r = re.compile(rf"(?<![\w]){re.escape(tok)}(?![\w])")
        WORD_BOUNDARY_CACHE[tok] = r
    return r

def token_present(text_lc: str, tok: str) -> bool:
    return bool(_word_re(tok).search(text_lc or ""))

def token_near_present(text_lc: str, tok: str, thresh: float = 0.90) -> bool:
    """
    Typo-tolerant whole-token check (no substrings).
    Only for alphabetic tokens with length >= 4 (avoid 'cat'→'application' noise).
    """
    if not tok.isalpha() or len(tok) <= 3:
        return False
    text_lc = (text_lc or "").lower()
    for w in tokenize(text_lc):
        if distance.Levenshtein.normalized_similarity(w, tok) >= thresh:
            return True
    return False

# --- Exact "section NNN" handling ---
SECTION_Q_RE = re.compile(r"\bsection\s+([0-9]+[a-z]?(?:\([0-9a-z]+\))?)\b", re.IGNORECASE)
 
def extract_section_number(q: str) -> str | None:
    """Return '412' or '412(4)' etc. from queries like 'section 412' (case-insensitive)."""
    m = SECTION_Q_RE.search(q or "")
    return m.group(1).lower() if m else None
 
def has_section_phrase(text_lc: str, sec_no: str) -> bool:
    """
    True if text contains an exact 'section 412' (flex spaces) OR 's. 412' variant.
    Uses token-safe boundaries to avoid substrings.
    text_lc is expected to be lowercase.
    """
    if not text_lc or not sec_no:
        return False
    # Accept "section 412" and "s. 412"
    patterns = [
        rf"section\s*{re.escape(sec_no)}",
        rf"s\.\s*{re.escape(sec_no)}",
    ]
    for pat in patterns:
        # word-boundary equivalent: no \w just before/after the whole phrase
        if re.search(rf"(?<![\w]){pat}(?![\w])", text_lc):
            return True
    return False

def fuzzy_score(a: str, b: str) -> float:
    """
    Single-token queries:
      - len <= 3: exact whole-word only (no substrings, no typos)
      - len >= 4: exact -> 1.0, else near-token -> 0.95
    Multi-token => normal fuzzy.
    """
    a = (a or "").lower()
    b = (b or "").lower()
    a_toks = tokenize(a)
 
    if len(a_toks) == 1:
        t = a_toks[0]
        L = len(t)
        if token_present(b, t):
            return 1.0
        if L >= 4 and token_near_present(b, t, thresh=0.90):
            return 0.95
        return 0.0
 
    return max(
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b),
        fuzz.ratio(a, b),
    ) / 100.0

# def fuzzy_score(a: str, b: str) -> float:
#     """
#     Robust fuzzy similarity.
#     For single-token queries, literal word presence inside a long text counts as a perfect hit.
#     """
#     a = (a or "").lower()
#     b = (b or "").lower()
#     a_toks = tokenize(a)
#     if len(a_toks) == 1:
#         if token_present(b, a_toks[0]):
#             return 1.0
#     return max(
#         fuzz.partial_ratio(a, b),
#         fuzz.token_set_ratio(a, b),
#         fuzz.ratio(a, b),
#     ) / 100.0

def get_embedding(text, model="text-embedding-3-small"):
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

# DF presence-based coverage 
LEADING_STOPWORDS = {
    "the","a","an","of","and","or","if","to","in","on","for","with","by",
    "is","are","be","was","were","at","from","as","that","this","these","those","it","about"
}

def corpus_presence_ratio(query: str) -> float:
    """
    Presence by DF: a token 'counts' if DF>=1 anywhere in the corpus.
    Ignore only leading trivial stopwords (keep dynamic ones eligible).
    """
    st = build_corpus_stats()
    toks = tokenize(query)
    # drop only leading trivial stopwords, not dynamic ones
    i = 0
    while i < len(toks) and toks[i] in LEADING_STOPWORDS:
        i += 1
    toks = toks[i:]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if st["df"].get(t, 0) >= 1)
    return hits / len(toks)

# --- light morphology + typo/prefix snap to vocab ---
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

def project_query_to_corpus(raw_query: str) -> str:
    st = build_corpus_stats()
    vocab, df = st["vocab"], st["df"]
    q_tokens = tokenize(raw_query)
    if not vocab or not q_tokens:
        return raw_query

    projected = []
    for tok in q_tokens:
        # Keep numeric tokens (and forms like 48(4)) intact; do not morph/correct them
        if tok.isdigit() or re.match(r'^\d+(?:\([0-9a-z]+\))?$', tok):
            projected.append(tok)
            continue

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

def corpus_coverage_ratio(query: str) -> float:
    st = build_corpus_stats()
    q_tokens = [t for t in tokenize(query) if t not in st["stop"]]
    if not q_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in st["vocab"])
    return hits / len(q_tokens)

def doc_hybrid_score(q, q_emb, d, name_like=False, q_tokens=None):
    title = (d.get("title") or "").lower()
    content = (d.get("content") or "").lower()
    emb = d.get("embedding")

    cos = float(cosine_similarity([q_emb], [emb])[0][0]) if (q_emb is not None and emb is not None) else 0.0
    f_title = fuzzy_score(q.lower(), title)
    f_cont  = fuzzy_score(q.lower(), content)

    pres = 0.0
    if name_like and q_tokens:
        hits = 0
        for t in q_tokens:
            if token_present(title, t) or token_present(content, t):
                hits += 1
        if hits:
            pres = min(1.0, hits / max(1, len(q_tokens)))

    return (0.70 * cos + 0.15 * f_title + 0.15 * f_cont + 0.18 * pres), cos
#pres-checks how many of the query tokens literally appear as whole words in the doc's title or content

# --- Leading stopword gate (UI typing nicety) ---
# NOTE: include digits so "839" is kept during quick typing checks
_WORDS_ANY = re.compile(r"[a-z0-9\(\)]+")

def _tokens_all(s: str):
    return _WORDS_ANY.findall((s or "").lower())

def _strip_leading_stopwords(tokens):
    i = 0
    while i < len(tokens) and tokens[i] in LEADING_STOPWORDS:
        i += 1
    return tokens[i:], i

# --- "X vs Y" title booster ---
VS_RE = re.compile(r"\b([a-z][a-z\-]{2,})\s+v(?:s\.?)?\s+([a-z][a-z\-]{1,})\b")

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

def _norm_title_text(s: str) -> str:
    return " ".join(tokenize(s or ""))

def _title_hits_general(raw_q: str, docs, top_n=5, min_score=0.80):
    qn = _norm_title_text(raw_q)
    if not qn:
        return []
    hits = []
    for d in docs:
        tn = _norm_title_text(d.get("title") or "")
        s = max(fuzz.token_set_ratio(qn, tn), fuzz.partial_ratio(qn, tn)) / 100.0
        if s >= min_score:
            hits.append((s, d))
    hits.sort(key=lambda x: x[0], reverse=True)

    out = []
    for s, d in hits[:top_n]:
        text = (d.get("content") or "").strip()
        chunk = split_paragraphs(text)
        if not chunk:
            ss = split_sentences(text)
            chunk = ss[:1] if ss else []
        snippet = (chunk[0] if chunk else "")[:400]
        out.append({
            "text": snippet,
            "title": d.get("title") or "",
            "url": d.get("file_url") or "",
            "score": s + 0.25
        })
    return out

def _title_hits_for_names(raw_q: str, docs, top_n=5, min_each=0.70, min_avg=0.78):
    st = build_corpus_stats()
    toks = [t for t in tokenize(raw_q) if t not in st["stop"]]
    if len(toks) < 2:
        toks = tokenize(raw_q)
    seen = set()
    q_tokens = [t for t in toks if not (t in seen or seen.add(t))]
    if len(q_tokens) < 2:
        return []

    hits = []
    for d in docs:
        title_tokens = tokenize(d.get("title") or "")
        if not title_tokens:
            continue
        per_tok = []
        for qt in q_tokens:
            best = 0.0
            for tt in title_tokens:
                best = max(best, fuzz.partial_ratio(qt, tt) / 100.0, fuzz.ratio(qt, tt) / 100.0)
            per_tok.append(best)
        good_sorted = sorted(per_tok, reverse=True)
        topk = good_sorted[:max(2, min(3, len(good_sorted)))]
        if len([s for s in topk if s >= min_each]) >= 2:
            avg_s = sum(topk) / len(topk)
            if avg_s >= min_avg:
                hits.append((avg_s + 0.28, d))

    hits.sort(key=lambda x: x[0], reverse=True)

    out = []
    for s, d in hits[:top_n]:
        text = (d.get("content") or "").strip()
        chunk = split_paragraphs(text)
        if not chunk:
            ss = split_sentences(text)
            chunk = ss[:1] if ss else []
        snippet = (chunk[0] if chunk else "")[:400]
        out.append({
            "text": snippet,
            "title": d.get("title") or "",
            "url": d.get("file_url") or "",
            "score": s
        })
    return out

#  /suggest 
@app.route("/suggest", methods=["GET"])
def suggest():
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or 8)

    if len(q) < 2:
        return jsonify({"q": q, "suggestions": []})

    # typing nicety: ignore leading trivial stopwords only
    toks_all = _tokens_all(q)
    i = 0
    while i < len(toks_all) and toks_all[i] in LEADING_STOPWORDS:
        i += 1
    tail_tokens = toks_all[i:]
    if i > 0 and not tail_tokens:
        return jsonify({"q": q, "suggestions": []})

    q_base = " ".join(tail_tokens) if tail_tokens else q

    alpha_count = sum(ch.isalpha() for ch in q_base)
    has_digit   = any(ch.isdigit() for ch in q_base)
    has_vs      = bool(VS_RE.search(q_base.lower()))  # e.g., "Gunasena v Bandarathilake"

    if alpha_count < 3 and not has_digit and not has_vs:
        return jsonify({"q": q, "suggestions": []})

    docs = load_docs_cache()

# If the query is like "section 412", extract the target section number
    sec_no = extract_section_number(q_base)  # e.g., "412" or "412(4)" or None

##########
########### SINGLE-TOKEN QUERIES (strict for short tokens; typo-tolerant for long names)
    q_tokens_all = tokenize(q_base)
    is_single_token = (len(q_tokens_all) == 1) and q_tokens_all[0].isalpha()
 
    if is_single_token:
        tok = q_tokens_all[0]
        L = len(tok)
        def _lc(x): return (x or "").lower()
 
        # 1) exact whole-word hits only
        keep = [
            d for d in docs
            if token_present(_lc(d.get("title")), tok) or token_present(_lc(d.get("content")), tok)
        ]
 
        if not keep:
            if L >= 4:
                # 2) try projecting token to corpus (handles common typos/morphs)
                proj = project_query_to_corpus(tok)
                ptoks = tokenize(proj)
                if ptoks:
                    pt = ptoks[0]
                    if pt != tok:
                        keep = [
                            d for d in docs
                            if token_present(_lc(d.get("title")), pt) or token_present(_lc(d.get("content")), pt)
                        ]
                # 3) as last resort for long tokens, allow near-token matches (no substrings)
                if not keep:
                    keep = [
                        d for d in docs
                        if token_near_present(_lc(d.get("title")), tok) or token_near_present(_lc(d.get("content")), tok)
                    ]
            else:
                # len <= 3 (e.g., "cat"): be strict — no projection, no near-token
                keep = []
 
        docs = keep
        if not docs:
            return jsonify({"q": q, "suggestions": []})
        

    # Title boosters are still computed for guards, but not used to pre-bias results
    vs_suggestions = _title_hits_for_vs(q_base, docs, top_n=1)
    # (we don't merge name/title lists anymore)

    # Project into corpus space
    q_proj = project_query_to_corpus(q_base)

    # Stats
    st = build_corpus_stats()
    content_terms = [t for t in tokenize(q_proj) if (t not in st["stop"] and t in st["vocab"])]
    key_q = " ".join(content_terms) if content_terms else q_proj.lower()

    orig_content_terms = [t for t in tokenize(q_base) if (t not in st["stop"] and t in st["vocab"])]
    orig_key_q = " ".join(orig_content_terms) if orig_content_terms else q_base.lower()

    vocab_only_terms = [t for t in tokenize(q_proj) if t in st["vocab"]]
    vocab_key = " ".join(vocab_only_terms) if vocab_only_terms else key_q

    # Coverage + presence
    coverage = corpus_coverage_ratio(q_proj) if len(q_proj) > 3 else 1.0
    out_of_corpus = (coverage == 0.0 and not vs_suggestions)

    q_tokens_all = tokenize(q_base)
    name_like = (len(q_tokens_all) <= 3)

    # Hard OOD block for short queries with zero DF presence (keeps Batman/Superman out)
    if q_tokens_all:
        presence = corpus_presence_ratio(q_proj)
        if presence == 0.0 and name_like and not vs_suggestions:
            logging.info("Suggest blocked by DF presence guard: q=%r tokens=%r", q, q_tokens_all)
            return jsonify({"q": q, "suggestions": []})

    # thresholds 
    if out_of_corpus and len(content_terms) <= 3:
        fuzzy_cut = 0.72  # min fuzzy match score a snippet must reach to be accepted
        doc_cos_cut = 0.18
    else:
        fuzzy_cut = 0.80 if out_of_corpus else 0.70
        doc_cos_cut = 0.30 if out_of_corpus else 0.15

    if name_like:
        doc_cos_cut = min(doc_cos_cut, 0.05)

    # Embed
    q_eff = key_q if len(key_q) >= len(orig_key_q) else orig_key_q
    q_emb = None
    if len(q_eff) >= 4:
        try:
            q_emb = get_embedding(q_eff)
        except Exception:
            q_emb = None

    # Preselect using your hybrid doc score (keeps behavior the same up to here)
    scored = []
    for d in docs:
        total, cos = doc_hybrid_score(q_eff, q_emb, d, name_like=name_like, q_tokens=q_tokens_all)
        scored.append((total, cos, d))
    scored.sort(key=lambda x: x[0], reverse=True)

    # ---------- Unified scoring: pick the stronger of Title vs Content for each doc ----------
    candidates = []
    qn = _norm_title_text(q_base)
    long_query = (len(tokenize(q_base)) >= 6) or (len(q_base) >= 30)
    q_lc = q_base.lower()

    # anchor terms = projected tokens; include numbers even if short
    has_digit = lambda s: any(c.isdigit() for c in s)
    # keep non-stop vocab tokens (len>=4) + any numeric-looking tokens from the projected query
    numeric_terms = [t for t in tokenize(q_proj) if has_digit(t)]
    if is_single_token:
        base_anchors = [q_tokens_all[0]]  # allow 3-letter single token (e.g., "cat") as an anchor
    else:
        base_anchors = [t for t in content_terms if len(t) >= 4]

    # dedupe while preserving order
    seen_anchor = set()
    anchor_terms = []
    for t in base_anchors + numeric_terms:
        if t not in seen_anchor:
            seen_anchor.add(t)
            anchor_terms.append(t)
    anchor_terms = anchor_terms[:6]  # keep it tight

    def _count_hits(text_lc: str, toks: list[str]) -> int:
        return sum(1 for t in toks if token_present(text_lc, t))

    pool_size = max(DOC_TOP_N * 3, 30)
    for total, d_cos, d in scored[:pool_size]:
        title = (d.get("title") or "")
        title_lc = title.lower()

        # Title similarity (normalized)
        title_f = fuzzy_score(qn, _norm_title_text(title))
        title_exact = False
        if q_tokens_all:
            tok_hits = sum(1 for t in q_tokens_all if token_present(title_lc, t))
            title_exact = (tok_hits >= max(1, len(q_tokens_all) - 1))
        if VS_RE.search(q_base.lower()):
            title_exact = title_exact or (title_f >= 0.90)

        # Content: find best snippet
        text = d.get("content") or ""
        chunks = split_paragraphs(text)
        if len(chunks) < 2:
            chunks = split_sentences(text)

        best_chunk = ""
        best_f = 0.0
        exact_chunk = False
        anchored_chunk = False

        for chunk in chunks:
            chunk_lc = chunk.lower()

            # If the query is "section NNN", hard-prefer chunks that contain that exact phrase
            if sec_no and has_section_phrase(chunk_lc, sec_no):
                f = 1.0
                exact_chunk = True
                anchored_chunk = True

            # literal hits on anchor terms (e.g., 'bulls', '839', '48(4)')
            hits_here = _count_hits(chunk_lc, anchor_terms)
            if hits_here >= 1:
                anchored_chunk = True

            f = max(
                fuzzy_score(key_q,      chunk_lc),
                fuzzy_score(orig_key_q, chunk_lc),
                fuzzy_score(vocab_key,  chunk_lc),
            )

            # exact long-phrase boost when present (use custom boundaries instead of \b)
            if long_query and re.search(rf"(?<![\w]){re.escape(q_lc)}(?![\w])", chunk_lc):
                f = max(f, 0.99)
                exact_chunk = True

            local_cut = fuzzy_cut
            # if the document-level cosine is decent, be a little more lenient
            if q_emb is not None and d_cos >= 0.20:
                local_cut = min(local_cut, 0.60)
            # if we have an anchor hit in this chunk, relax a bit more
            if anchored_chunk:
                local_cut = min(local_cut, 0.65)

            if f < local_cut and not exact_chunk:
                continue

            if f > best_f:
                best_f = f
                best_chunk = chunk

        # Choose stronger channel (title vs content)
        channel_f = best_f
        snippet = best_chunk
        exact = exact_chunk

        if title_f >= best_f:
            channel_f = title_f
            exact = exact or title_exact
            if not snippet:
                ss = split_paragraphs(text) or split_sentences(text)
                snippet = ss[0] if ss else ""

        ###
            # --- Numeric anchor tweak (prioritise exact number hits) ---
        num_anchors = [t for t in anchor_terms if any(c.isdigit() for c in t)]
        num_hits = _count_hits((snippet or "").lower(), num_anchors)

        # Final score: mostly similarity, backed by doc cosine
        # Final score: mostly similarity, backed by doc cosine
        #####
        # --- PRIORITIZE exact "section NNN" matches in title or content ---
        sec_exact_doc = False
        if sec_no:
            title_lc_full = title_lc  # already lower
            content_lc_full = (d.get("content") or "").lower()
            if has_section_phrase(title_lc_full, sec_no) or has_section_phrase(content_lc_full, sec_no):
                sec_exact_doc = True
                exact = True  # treat as exact for ranking purposes
                # If our chosen snippet didn't include the phrase, still give it high confidence
                channel_f = max(channel_f, 0.99)
 
        # Final score: mostly similarity, backed by doc cosine; weight content more if digits involved
        content_w = 0.5 if long_query else 0.4
        if num_anchors:
            content_w = 0.7  # bias more toward content when numbers are in the query
 
        score = (content_w * channel_f + (1 - content_w) * max(d_cos, 0.0))
        if exact:
            score += 0.12
        if num_hits:
            score += 0.20
        if sec_exact_doc:
            score += 1.20  # BIG bump to put exact "section 412" docs at the very top
        # final gate: normally 0.78, but allow entries that have an anchor hit + reasonable cosine
        base_gate = 0.78
        if long_query:
            base_gate -= 0.03  # a touch more forgiving for long queries
        if num_anchors:
            base_gate -= 0.05  # slightly easier gate when digit anchors are present

        # did any (alpha or numeric) anchor term appear in the chosen snippet?
        anchored_final = _count_hits((snippet or "").lower(), anchor_terms) >= 1

        passes_gate = (channel_f >= base_gate) or (anchored_final and d_cos >= 0.12)
        #
        if passes_gate:
            candidates.append({
                "text": (snippet or "")[:400],
                "title": d.get("title") or "",
                "url": d.get("file_url") or "",
                "score": score
            })

    # Sort best-first and truncate
    merged = sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]


    logging.info(
        "Suggest q=%r base=%r proj=%r key=%r coverage=%.2f OOD=%s name_like=%s n=%d",
        q, q_base, q_proj, key_q, coverage, out_of_corpus, name_like, len(merged)
    )
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
