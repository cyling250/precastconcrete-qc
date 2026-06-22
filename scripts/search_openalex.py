"""Search OpenAlex for papers on point cloud segmentation of reinforced concrete."""
import json, urllib.request, sys

PROXY = "http://127.0.0.1:7897"

def search(query: str, per_page: int = 20):
    url = f"https://api.openalex.org/works?search={query}&filter=type:article,publication_year:2019-2025&sort=cited_by_count:desc&per_page={per_page}"
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    with opener.open(url) as resp:
        return json.loads(resp.read())

# Query 1: point cloud segmentation + reinforced concrete
print("=" * 110)
print("QUERY: point cloud semantic segmentation reinforced concrete")
print("=" * 110)
d = search("point%20cloud%20semantic%20segmentation%20reinforced%20concrete")
for i, r in enumerate(d.get("results", []), 1):
    src = r.get("primary_location", {}) or {}
    journal = (src.get("source") or {}).get("display_name", "N/A") if src.get("source") else "N/A"
    print(f"{i:2d}. [{r['cited_by_count']:3d} cit] {r['publication_year']} | {journal[:35]:35s} | {r['title'][:90]}")

print()
print("=" * 110)
print("QUERY: rebar detection point cloud deep learning")
print("=" * 110)
d = search("rebar%20detection%20point%20cloud%20deep%20learning")
for i, r in enumerate(d.get("results", []), 1):
    src = r.get("primary_location", {}) or {}
    journal = (src.get("source") or {}).get("display_name", "N/A") if src.get("source") else "N/A"
    print(f"{i:2d}. [{r['cited_by_count']:3d} cit] {r['publication_year']} | {journal[:35]:35s} | {r['title'][:90]}")

print()
print("=" * 110)
print("QUERY: precast concrete component point cloud segmentation")
print("=" * 110)
d = search("precast%20concrete%20component%20point%20cloud%20segmentation")
for i, r in enumerate(d.get("results", []), 1):
    src = r.get("primary_location", {}) or {}
    journal = (src.get("source") or {}).get("display_name", "N/A") if src.get("source") else "N/A"
    print(f"{i:2d}. [{r['cited_by_count']:3d} cit] {r['publication_year']} | {journal[:35]:35s} | {r['title'][:90]}")

print()
print("=" * 110)
print("QUERY: building element point cloud segmentation synthetic data")
print("=" * 110)
d = search("building%20element%20point%20cloud%20segmentation%20synthetic%20data")
for i, r in enumerate(d.get("results", []), 1):
    src = r.get("primary_location", {}) or {}
    journal = (src.get("source") or {}).get("display_name", "N/A") if src.get("source") else "N/A"
    print(f"{i:2d}. [{r['cited_by_count']:3d} cit] {r['publication_year']} | {journal[:35]:35s} | {r['title'][:90]}")
