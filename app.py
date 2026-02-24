from pathlib import Path
import sys
import importlib.util

# Ensure backend/ is on sys.path so backend modules like rag_graph and rag_store
# can be imported as top-level modules when backend/app.py runs.
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

spec = importlib.util.spec_from_file_location("backend_app", BACKEND / "app.py")
backend_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_app)

# Expose the ASGI app object for uvicorn: `uvicorn app:app`
app = getattr(backend_app, "app")
