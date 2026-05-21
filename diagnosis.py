from collections import defaultdict
import re
from pathlib import Path

site_counts = defaultdict(set)

for p in Path("/data/raw/images").rglob("*.tiff"):
    m = re.search(r"(r\d{2}c\d{2})", p.name.lower())
    s = re.search(r"f(\d{2})p\d{2}", p.name.lower())

    if m and s:
        well = m.group(1)
        site = int(s.group(1))
        site_counts[well].add(site)

# summarize
lengths = [len(v) for v in site_counts.values()]

print("min sites per well:", min(lengths))
print("max sites per well:", max(lengths))
print("unique patterns:", set(lengths))