import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import settings  
from rag.corpus import download_sample_papers  
from rag.llm import OllamaClient


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--no-download"]
    if "--no-download" not in sys.argv:
        print("Checking sample papers in data/documents …")
        for name, status, msg in download_sample_papers(settings.docs_dir):
            print(f"  [{status:>10}] {name}  {msg if status != 'present' else ''}")
    ok, reason = OllamaClient(settings.ollama_url, settings.llm_model).status()
    print(f"LLM: {reason}")
    if not ok:
        print("  The app still works in extractive mode. For generated answers install Ollama "
              f"(https://ollama.com) and run: ollama pull {settings.llm_model}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
           "--browser.gatherUsageStats=false", *args]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
