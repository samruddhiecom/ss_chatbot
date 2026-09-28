from dotenv import load_dotenv
load_dotenv()
import os
import notion_sync
notion_sync.NOTION_TOKEN = os.environ["NOTION_TOKEN"]
notion_sync.NOTION_DB_ID = os.environ["NOTION_DB_ID"]
chunks = notion_sync.fetch_chunks()
print("Fetched", len(chunks), "chunks from Notion")
for meta, text in chunks:
    print(f"  [{meta[chr(39)+chr(108)+chr(97)+chr(121)+chr(101)+chr(114)+chr(39)]}] {meta[chr(39)+chr(110)+chr(97)+chr(109)+chr(101)+chr(39)]}")
