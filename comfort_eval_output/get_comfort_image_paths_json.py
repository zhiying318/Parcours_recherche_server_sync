import json
from pathlib import Path

root = Path("./COMFORT/data/comfort_human_car")

# 只搜索两级子目录下的 .png 文件
image_paths = [
    str(p) for p in root.glob("*/*/*.png")
]

image_paths.sort()
print(f"Found {len(image_paths)} images")

with open("comfort_image_paths.json", "w", encoding="utf-8") as f:
    json.dump(image_paths, f, indent=2, ensure_ascii=False)

print("Saved: comfort_image_paths.json")