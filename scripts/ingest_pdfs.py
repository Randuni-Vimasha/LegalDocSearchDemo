# scripts/ingest_pdfs.py
import os, re
from supabase import create_client
from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv
import base64, io, time

load_dotenv()

# ======= HARD-CODED KEYS =======
# ===========================================
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
BUCKET_NAME    = os.getenv("BUCKET_NAME", "legal-pdfs")  #

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OpenAI_API_KEY)

#  
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ...existing code...
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdf"))
# ...existing code...

# Storage upload options MUST be strings for this SDK/httpx
FILE_OPTS = {
    "upsert": "true",
    "content-type": "application/pdf",
    "cache-control": "3600",
}

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()  # collapse whitespace
    return t

def get_embedding(text: str, model="text-embedding-ada-002"):
    return oa.embeddings.create(model=model, input=text).data[0].embedding

def in_bucket(filename: str) -> bool:
    # check root of bucket for the file
    try:
        items = sb.storage.from_(BUCKET_NAME).list()
        return any(i.get("name") == filename for i in items)
    except Exception:
        return False

def public_url(filename: str) -> str:
    return sb.storage.from_(BUCKET_NAME).get_public_url(filename)

def pdf_pages_to_b64_images(pdf_path: str, dpi: int = 300, max_dim: int = 1400):
    """
    Render each page to PNG bytes (via PyMuPDF), optionally downscale to cap tokens,
    and return a list of base64 strings ('data:image/png;base64,...' without prefix).
    """
    b64s = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)        # render
            png_bytes = pix.tobytes("png")        # PNG in-memory
            
            
            b64s.append(base64.b64encode(png_bytes).decode("utf-8"))
    return b64s

def ocr_pdf_with_openai(pdf_path: str, model: str = None):
    """
    OCR every page with OpenAI (vision). Returns full_text and elapsed seconds.
    """
    mdl = model or os.getenv("OCR_MODEL", "gpt-4o-mini")
    images = pdf_pages_to_b64_images(pdf_path, dpi=300)
    parts = []
    t0 = time.time()
    for i, b64 in enumerate(images, 1):
        try:
            resp = oa.chat.completions.create(
                model=mdl,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Extract all readable text from this page. Return plain text only."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ],
                }],
                temperature=0
            )
            page_text = (resp.choices[0].message.content or "").strip()
            parts.append(f"\n--- Page {i} ({mdl}) ---\n{page_text}")
        except Exception as e:
            parts.append(f"\n--- Page {i} ({mdl}) ---\n[OCR failed: {e}]")
    return "\n".join(parts), time.time() - t0

def likely_scanned(text: str, min_chars: int = 200):

    t = re.sub(r"\s+", " ", (text or "")).strip()
    return len(t) < min_chars


def main():
    pdfs = [f for f in os.listdir(PROJECT_ROOT) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in:", PROJECT_ROOT)
        return

    for filename in pdfs:
        local_path = os.path.join(PROJECT_ROOT, filename)

        # Skip if same title already in DB
        if sb.table("legal_documents").select("id").eq("title", filename).execute().data:
            print("DB already has:", filename, "— skipping row insert.")
            # ensure Storage has it (upload once if missing)
            if not in_bucket(filename):
                with open(local_path, "rb") as f:
                    sb.storage.from_(BUCKET_NAME).upload(filename, f, FILE_OPTS)
            continue

        # Upload to Storage if not there
        if not in_bucket(filename):
            with open(local_path, "rb") as f:
                sb.storage.from_(BUCKET_NAME).upload(filename, f, FILE_OPTS)
            print("Uploaded to Storage:", filename)
        else:
            print("Already in Storage:", filename)

        url = public_url(filename)

        # Extract text
        with fitz.open(local_path) as pdf:
            raw_text = " ".join(page.get_text() for page in pdf)

        # If there’s almost no text layer, run GPT-4o-mini OCR
        if likely_scanned(raw_text):
            print("No/low text layer → running OpenAI OCR:", filename)
            ocr_text, secs = ocr_pdf_with_openai(local_path)

            text = ocr_text
        else:
            text = raw_text


        # Clean + embed
        cleaned = clean_text(text)
        try:
            emb = get_embedding(cleaned)
        except Exception as e:
            print("Embedding failed:", filename, e)
            continue

        # Insert row (embedding column is vector → send list)
        sb.table("legal_documents").insert({
            "title": filename,
            "content": text,
            "embedding": emb,
            "file_url": url
        }).execute()

        print("Inserted DB row:", filename)

if __name__ == "__main__":
    main()
