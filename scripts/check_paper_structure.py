"""Search top point cloud / construction papers for their Methodology structure."""
import json, urllib.request, urllib.parse

PROXY = "http://127.0.0.1:7897"

def search(query: str, per_page: int = 5):
    q = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q}&filter=type:article,publication_year:2019-2025&sort=cited_by_count:desc&per_page={per_page}"
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    with opener.open(url) as resp:
        return json.loads(resp.read())

# PointNet++ — the original paper
print("=" * 100)
print("PointNet++, Qi et al. 2017 — Section structure")
print("=" * 100)
print("""
3. PointNet++
  3.1 Review of PointNet
  3.2 Hierarchical Point Set Feature Learning
     (Set Abstraction levels: Sampling, Grouping, PointNet)
  3.3 Robust Feature Learning under Non-Uniform Density
     (Multi-scale grouping / MRG)
  3.4 Point Feature Propagation for Segmentation
""")

# Random Forest / construction papers structure
print("=" * 100)
print("Typical Construction Engineering + DL paper structure (e.g., AutoCon, CACIE)")
print("=" * 100)
print("""
Methodology (or Proposed Method)
  2.1 Overview of the Proposed Framework
  2.2 Data Acquisition and Preprocessing
  2.3 xxx Detection / Segmentation Algorithm
    2.3.1 Network Architecture
    2.3.2 Loss Function
    2.3.3 Training Strategy
  2.4 Evaluation Metrics
""")

print("=" * 100)
print("Synthetic point cloud for construction (Zou et al. 2025, CACIE)")
print("=" * 100)
d = search("synthetic point cloud segmentation building structure", per_page=1)
for r in d.get("results", []):
    print(f"[{r['publication_year']}] {r['title'][:80]}")

print()
print("=" * 100)
print("Point Transformer, Zhao et al. 2021")
print("=" * 100)
d = search("point transformer point cloud segmentation", per_page=1)
for r in d.get("results", []):
    print(f"[{r['publication_year']}] {r['title'][:80]}")
print("""
3. Method
  3.1 Background (Point Cloud Representation)
  3.2 Point Transformer Layer
     (Attention, Vector Attention, Position Encoding)
  3.3 Network Architecture
     (Encoder, Decoder, Skip Connections)
  3.4 Implementation Details
""")
