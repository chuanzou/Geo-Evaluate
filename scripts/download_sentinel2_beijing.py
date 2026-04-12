from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


BEIJING_AOI = [115.4, 39.4, 117.6, 41.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Google Earth Engine export tasks for cloud-free Sentinel-2 Beijing patches."
    )
    parser.add_argument("--year", type=int, default=2023, help="Year to export.")
    parser.add_argument("--start-date", default=None, help="Override start date, e.g. 2023-04-01.")
    parser.add_argument("--end-date", default=None, help="Override end date, e.g. 2023-10-31.")
    parser.add_argument("--num-patches", type=int, default=2000, help="Number of random patches to export.")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch width/height in pixels.")
    parser.add_argument("--scale", type=int, default=10, help="Export scale in meters.")
    parser.add_argument("--max-scene-cloud", type=float, default=10.0, help="Max scene CLOUDY_PIXEL_PERCENTAGE.")
    parser.add_argument("--drive-folder", default="GeoEvaluate_Beijing_S2", help="Google Drive export folder.")
    parser.add_argument("--description-prefix", default="s2_beijing_patch", help="Earth Engine task name prefix.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patch centers.")
    parser.add_argument("--max-tasks", type=int, default=300, help="Safety cap for tasks started in one run.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned exports without starting tasks.")
    return parser.parse_args()


def import_ee():
    try:
        import ee
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: earthengine-api. Install it with `python -m pip install '.[geo]'` "
            "or `python -m pip install earthengine-api`."
        ) from exc
    return ee


def initialize_ee(ee) -> None:
    try:
        ee.Initialize()
    except Exception:
        print("Earth Engine is not initialized. Starting authentication flow...")
        ee.Authenticate()
        ee.Initialize()


def mask_sentinel2_sr(ee, image):
    scl = image.select("SCL")
    valid_scl = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    optical = image.select(["B4", "B3", "B2", "B8"], ["red", "green", "blue", "nir"])
    return optical.updateMask(valid_scl).divide(10000).copyProperties(image, ["system:time_start"])


def build_composite(ee, args: argparse.Namespace):
    start_date = args.start_date or f"{args.year}-04-01"
    end_date = args.end_date or f"{args.year}-10-31"
    aoi = ee.Geometry.Rectangle(BEIJING_AOI, proj="EPSG:4326", geodesic=False)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", args.max_scene_cloud))
        .map(lambda image: mask_sentinel2_sr(ee, image))
    )
    composite = collection.median().clip(aoi).toFloat()
    return aoi, composite, start_date, end_date


def main() -> None:
    args = parse_args()
    ee = import_ee()
    initialize_ee(ee)

    if args.num_patches > args.max_tasks:
        raise SystemExit(
            f"Refusing to start {args.num_patches} tasks because --max-tasks is {args.max_tasks}. "
            "Raise --max-tasks deliberately if you really want that many exports in one batch."
        )

    aoi, composite, start_date, end_date = build_composite(ee, args)
    patch_meters = args.patch_size * args.scale
    half_patch = patch_meters / 2
    points = ee.FeatureCollection.randomPoints(
        region=aoi,
        points=args.num_patches,
        seed=args.seed,
        maxError=100,
    ).toList(args.num_patches)

    print(
        f"Preparing {args.num_patches} Sentinel-2 patch exports for Beijing "
        f"from {start_date} to {end_date}."
    )
    print(f"Patch size: {args.patch_size}x{args.patch_size} px at {args.scale} m.")
    print(f"Drive folder: {args.drive_folder}")

    for index in range(args.num_patches):
        point = ee.Feature(points.get(index)).geometry()
        region = point.buffer(half_patch).bounds(maxError=10)
        prefix = f"{args.description_prefix}_{args.year}_{index:05d}"
        if args.dry_run:
            print(f"[dry-run] {prefix}")
            continue

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=prefix,
            folder=args.drive_folder,
            fileNamePrefix=prefix,
            region=region,
            scale=args.scale,
            crs="EPSG:3857",
            maxPixels=1_000_000,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        print(f"Started task {index + 1}/{args.num_patches}: {prefix}")
        time.sleep(0.15)

    print("Done. Monitor tasks in the Earth Engine Tasks tab or with the Earth Engine CLI.")
    print("After GeoTIFFs arrive in Drive, download them and convert each 4-band file to .npy if needed.")


if __name__ == "__main__":
    main()
