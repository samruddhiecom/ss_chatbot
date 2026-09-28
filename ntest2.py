import os
from notion_client import Client

TOKEN = os.environ.get("NOTION_TOKEN", "")
DB_ID = os.environ.get("NOTION_DB_ID", "")
nc = Client(auth=TOKEN)

db = nc.databases.retrieve(DB_ID)
print("db keys:", list(db.keys()))
ds = db.get("data_sources", [])
print("data_sources:", ds)

if ds:
    dsid = ds[0]["id"]
    print("querying data source:", dsid)
    res = nc.request(path=f"data_sources/{dsid}/query", method="POST", body={"page_size": 20})
else:
    res = nc.request(path=f"databases/{DB_ID}/query", method="POST", body={"page_size": 20})

rows = res.get("results", [])
print("ROWS FOUND:", len(rows))
for r in rows:
    props = r.get("properties", {})
    name = ""
    for k, v in props.items():
        if v.get("type") == "title":
            name = "".join(x.get("plain_text","") for x in v.get("title", []))
    print("  -", name)
print("DONE")