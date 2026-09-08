import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv()
import os
import notion_sync
notion_sync.NOTION_TOKEN = os.environ["NOTION_TOKEN"]
notion_sync.NOTION_DB_ID = os.environ["NOTION_DB_ID"]
from app.kb import store

def header(meta):
    return f"[Layer {meta.get('layer','?')} | stage: {meta.get('stage','any')} | {meta.get('type','guidance')}]"

chunks = notion_sync.fetch_chunks()
store.reset_collection()
ids, docs, metas = [], [], []
excluded = 0
for i, (meta, text) in enumerate(chunks):
    layer = meta.get("layer","").upper()
    if layer == "C":
        excluded += 1
        continue
    docs.append(f"{header(meta)} {text}")
    ids.append(f"{layer}-{meta.get('stage','any')}-{meta.get('type','g')}-{i}")
    metas.append({"layer": layer, "stage": meta.get("stage","any"), "type": meta.get("type","guidance"), "source": "notion", "name": meta.get("name","")})

store.add(ids, docs, metas)
print(f"Ingested {len(docs)} chunks from Notion into ChromaDB")
print(f"Layer C excluded (by design): {excluded}")
print(f"Collection now holds: {store.count()} chunks")
