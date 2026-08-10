#!/usr/bin/env python3
"""Overlay the official Blender orientation axes on saved batch predictions."""

import argparse
import json
import math
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/coco_smoke/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coco_smoke/visualizations"))
    parser.add_argument(
        "--renderer",
        choices=("pil", "blender"),
        default="pil",
        help="PIL works without system GUI libraries; Blender uses the official 3-D axes.",
    )
    return parser.parse_args()


def resolve_image(path_text):
    path = Path(path_text)
    if path.is_file():
        return path
    marker = "Orient-Anything-V2/"
    normalized = path_text.replace("\\", "/")
    container_marker = "/workspace/project/"
    if container_marker in normalized:
        candidate = WORKSPACE_ROOT / normalized.split(container_marker, 1)[1]
        if candidate.is_file():
            return candidate
    if marker in normalized:
        candidate = PROJECT_ROOT / normalized.split(marker, 1)[1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot map result image path to this checkout: {path_text}")


def add_caption(image, row):
    canvas = Image.new("RGB", (image.width, image.height + 70), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, image.height + 8), Path(row["image"]).name, fill="black")
    draw.text(
        (10, image.height + 32),
        "az={:.0f}°  el={:.0f}°  rot={:.0f}°  directions={}".format(
            row["azimuth_deg"],
            row["elevation_deg"],
            row["in_plane_rotation_deg"],
            row["num_front_directions"],
        ),
        fill="black",
    )
    return canvas


def render_pil_indicator(source, row):
    """Draw an azimuth compass without Blender or X11 system libraries."""
    overlay = source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(overlay)
    center = (442, 70)
    radius = 52
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        fill=(255, 255, 255), outline=(0, 0, 0), width=3,
    )
    azimuth = math.radians(float(row["azimuth_deg"]))
    endpoint = (
        center[0] + round(radius * 0.78 * math.sin(azimuth)),
        center[1] - round(radius * 0.78 * math.cos(azimuth)),
    )
    draw.line((center, endpoint), fill=(220, 25, 25), width=6)
    draw.ellipse((endpoint[0] - 5, endpoint[1] - 5, endpoint[0] + 5, endpoint[1] + 5), fill=(220, 25, 25))
    draw.text((center[0] - 4, center[1] - radius + 4), "0", fill="black")
    draw.text((center[0] + radius - 19, center[1] - 7), "90", fill="black")
    draw.text((center[0] - 10, center[1] + radius - 16), "180", fill="black")
    draw.text((center[0] - radius + 3, center[1] - 7), "270", fill="black")
    return overlay


def make_contact_sheet(images, output_path, columns=4, tile_width=300):
    thumbnails = []
    for image in images:
        thumb = image.copy()
        thumb.thumbnail((tile_width, tile_width + 50))
        tile = Image.new("RGB", (tile_width, tile_width + 80), "#dddddd")
        tile.paste(thumb, ((tile.width - thumb.width) // 2, 5))
        thumbnails.append(tile)
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_width + 80)), "#bbbbbb")
    for index, tile in enumerate(thumbnails):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(output_path, quality=92)


def main():
    args = parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if row.get("status") == "ok"]
    if not rows:
        raise SystemExit("No successful predictions to visualize.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    renderer = None
    if args.renderer == "blender":
        # Import lazily: bpy may require libXrender even though PIL rendering does not.
        from utils.axis_renderer import BlendRenderer
        renderer = BlendRenderer()
    rendered = []
    for index, row in enumerate(rows, start=1):
        # Dataset inference may use a human crop while retaining the original
        # path in `image`. Draw axes on the exact image given to the model.
        source_path = resolve_image(row.get("inference_image") or row["image"])
        source = Image.open(source_path).convert("RGB")
        if renderer is None:
            overlay = render_pil_indicator(source, row)
        else:
            with tempfile.NamedTemporaryFile(suffix=".png") as axis_file:
                renderer.render_axis(
                    row["azimuth_deg"],
                    row["elevation_deg"],
                    row["in_plane_rotation_deg"],
                    row["num_front_directions"],
                    axis_file.name,
                )
                axis = Image.open(axis_file.name).convert("RGBA")
                resized = source.resize(axis.size, Image.Resampling.LANCZOS)
                overlay = Image.alpha_composite(resized.convert("RGBA"), axis).convert("RGB")

        captioned = add_caption(overlay, row)
        output_path = args.output_dir / f"{index:02d}_{source_path.stem}_axes.jpg"
        captioned.save(output_path, quality=95)
        rendered.append(captioned)
        print(output_path)

    make_contact_sheet(rendered, args.output_dir / "contact_sheet.jpg")
    print(args.output_dir / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
