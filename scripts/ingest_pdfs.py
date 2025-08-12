# scripts/ingest_pdfs.py
import os, re
from supabase import create_client
from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()

# ======= HARD-CODED KEYS =======
# ===========================================
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY")
BUCKET_NAME    = os.getenv("BUCKET_NAME", "legal-pdfs")  #

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OpenAI_API_KEY)

# PDFs live one level up from this script (project root = LegalPdfs/)
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
            text = " ".join(page.get_text() for page in pdf)

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
