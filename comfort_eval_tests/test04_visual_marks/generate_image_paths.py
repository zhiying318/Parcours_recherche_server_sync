import json
from pathlib import Path

# Run from /home/zzou:
#   python comfort_eval_tests/test04_visual_marks/generate_image_paths.py

root = Path("./COMFORT/data/comfort_human_car_visual_marks")

image_paths = sorted(str(p) for p in root.glob("*/*/*.png"))
print(f"Found {len(image_paths)} images")

output = Path("comfort_eval_tests/test04_visual_marks/image_paths.json")
with open(output, "w", encoding="utf-8") as f:
    json.dump(image_paths, f, indent=2, ensure_ascii=False)

print(f"Saved: {output}")
