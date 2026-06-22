"""More targeted search for highly relevant papers."""
import json, urllib.request, urllib.parse

PROXY = "http://127.0.0.1:7897"

def search(query: str, per_page: int = 15, year_from: int = 2018):
    q = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q}&filter=type:article,publication_year:{year_from}-2025&sort=cited_by_count:desc&per_page={per_page}"
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    with opener.open(url) as resp:
        return json.loads(resp.read())

queries = [
    "automated rebar diameter classification point cloud",
    "precast concrete dimensional quality assessment point cloud deep learning",
    "segmentation structural elements point cloud deep learning building",
    "reinforcement steel bar detection 3D point cloud laser scanning",
    "synthetic point cloud semantic segmentation construction",
    "pointnet plus plus building infrastructure segmentation",
    "BIM point cloud semantic segmentation reinforced concrete",
]

for q in queries:
    print("=" * 110)
    print(f"QUERY: {q}")
    print("=" * 110)
    d = search(q, per_page=10)
    for i, r in enumerate(d.get("results", []), 1):
        src = r.get("primary_location", {}) or {}
        journal = (src.get("source") or {}).get("display_name", "N/A") if src.get("source") else "N/A"
        doi = (r.get("doi") or "N/A")[:50]
        abs_ = (r.get("abstract_inverted_index") is not None)
        print(f"{i:2d}. [{r['cited_by_count']:3d} cit] {r['publication_year']} | {journal[:35]:35s}")
        print(f"     {r['title'][:100]}")
        print(f"     DOI: {doi}")
        print()
