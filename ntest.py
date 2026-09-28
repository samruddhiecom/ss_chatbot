import os
from notion_client import Client

TOKEN = os.environ.get("NOTION_TOKEN", "")
DB_ID = os.environ.get("NOTION_DB_ID", "")
nc = Client(auth=TOKEN)

print("=== token ===")
try:
    me = nc.users.me()
    print("valid:", me.get("name"), me.get("type"))
except Exception as e:
    print("token error:", str(e)[:200]); raise SystemExit

print("=== db ===")
try:
    db = nc.databases.retrieve(DB_ID)
    print("props:", [f"{k}:{v.get('type')}" for k,v in db.get("properties",{}).items()])
except Exception as e:
    print("db error:", str(e)[:300]); raise SystemExit

print("=== rows ===")
res = nc.databases.query(database_id=DB_ID, page_size=20)
print("rows:", len(res.get("results",[])))
print("SUCCESS")