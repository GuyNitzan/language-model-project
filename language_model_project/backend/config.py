import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
QDRANT_URL        = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME   = os.environ.get("QDRANT_COLLECTION", "dermatology_book")

EMBED_MODEL  = "text-embedding-3-large"
CLAUDE_MODEL = "claude-sonnet-4-6"
TOP_K        = 8    # chunks to retrieve per query
MAX_HISTORY  = 10   # message pairs to keep in context
