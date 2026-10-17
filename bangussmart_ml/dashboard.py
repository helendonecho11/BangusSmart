"""Local dashboard records and explicitly bounded age estimates."""
import csv
import io
import json
import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field, ConfigDict
from bangussmart_ml import MODELS, ROOT, rule_status, STATUSES, feed_rate_pct, FEED_FACTOR, calculate_feed

router = APIRouter(prefix="/api")


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    avg_weight_g: float | None = Field(default=None, gt=0, le=10000)
    mean_temp_c: float = Field(default=29, ge=0, le=45)
    day: int | None = Field(default=None, ge=0, le=3650)
    stocking_date: date | None = None
    hatch_date: date | None = None
    observation_date: date
    fish_count: int = Field(default=5000, ge=0, le=10000000)
    temp_c: float | None = Field(default=None, ge=-5, le=50)
    ph: float | None = Field(default=None, ge=0, le=14)
    do_mgl: float | None = Field(default=None, ge=0, le=50)
    meals_per_day: int = Field(default=2, ge=1, le=12)
    feed_rate_pct: float | None = Field(default=None, gt=0, le=10)


@lru_cache(maxsize=1)
def growth_model():
    return joblib.load(MODELS / "growth_model.joblib")


def estimate_age(item: Observation):
    result = {"age_days": None, "culture_days": None, "method": "unavailable",
              "note": "A photo alone cannot determine age. Add a known date or measured weight."}
    for field, key in (("hatch_date", "age_days"), ("stocking_date", "culture_days")):
        value = getattr(item, field)
        if value:
            delta = (item.observation_date - value).days
            if delta < 0:
                raise ValueError(f"{field} must be on or before observation_date")
            result[key] = delta
    if item.hatch_date and item.stocking_date and item.hatch_date > item.stocking_date:
        raise ValueError("Hatch date must be on or before stocking date")
    if item.day is not None:
        if result["culture_days"] is not None and item.day != result["culture_days"]:
            raise ValueError("Day of culture conflicts with stocking and observation dates")
        result["culture_days"] = item.day
    if result["age_days"] is not None or result["culture_days"] is not None:
        result.update(method="recorded", note="Calculated from your records. Days of culture are days since stocking, not age since hatching.")
    elif item.avg_weight_g is not None:
        if not 26.5 <= item.mean_temp_c <= 31.5:
            result["note"] = "Weight estimation supports mean temperatures 26.5–31.5 °C only. Supply known dates."
            return result
        days = np.arange(0, 148)
        weights = growth_model().predict(pd.DataFrame({"day": days, "mean_temp_c": item.mean_temp_c}))
        if not weights.min() <= item.avg_weight_g <= weights.max():
            result["note"] = "Weight is outside the synthetic model's supported range at this temperature. Supply known dates."
            return result
        nearest = np.abs(weights - item.avg_weight_g)
        matches = days[np.isclose(nearest, nearest.min())]
        result.update(culture_days=int(round(float(np.median(matches)))), method="synthetic_weight_estimate",
                      matching_days=[int(matches.min()), int(matches.max())],
                      note="Rough days-since-stocking estimate by matching measured weight to a synthetic growth model (0–147 days). Not biological age; not validated on real fish. Matching days are model ties, not a confidence interval.")
    return result


def feeding_plan(item: Observation):
    missing = [label for key, label in (("avg_weight_g", "measured average weight"), ("temp_c", "water temperature"), ("ph", "pH"), ("do_mgl", "dissolved oxygen")) if getattr(item, key) is None]
    if missing:
        return {"available": False, "missing": missing, "note": "Add measured weight, fish count and complete water readings to calculate a feed plan. Photos and videos do not measure these inputs."}
    status = STATUSES[int(rule_status(item.temp_c, item.ph, item.do_mgl))]
    shared = calculate_feed(item.avg_weight_g, item.fish_count, status, rate_override=item.feed_rate_pct)
    rate, biomass, daily = shared['rate_pct'], shared['biomass_kg'], shared['daily_feed_kg']
    return {"available": True, "biomass_kg": round(biomass, 3), "fish_count": item.fish_count,
            "measured_weight_g": item.avg_weight_g, "rate_pct": rate,
            "rate_source": "user_supplied" if item.feed_rate_pct is not None else "prototype_placeholder_table", "rate_was_capped": shared['rate_was_capped'],
            "water_status": status, "water_multiplier": FEED_FACTOR[status],
            "daily_feed_kg": round(daily, 3), "meals_per_day": item.meals_per_day,
            "feed_per_meal_kg": round(daily / item.meals_per_day, 3),
            "note": "Prototype calculation, not a validated feeding prescription. Default feeding rates and water reductions are assumptions. Confirm the ration, feed product, pellet size and schedule with your farm adviser or feed supplier. Equal portions are a calculation only."}


def analyze_observation(item: Observation):
    return {**estimate_age(item), "feeding": feeding_plan(item),
            "vision": {"trained_model_available": False, "age_from_media": None,
                       "weight_from_media": None, "species_verified": False}}


@router.post("/age")
def age(item: Observation):
    try:
        return analyze_observation(item)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Growth model missing. Train the prototype first.") from exc


class PhotoUpload(BaseModel):
    name: str = Field(max_length=200)
    image: str = Field(max_length=12_000_000)
    observation: Observation


@router.post("/photos")
def photo_upload(item: PhotoUpload):
    import base64
    from uuid import uuid4
    from PIL import Image, UnidentifiedImageError
    try:
        binary = base64.b64decode(item.image.split(",", 1)[-1], validate=True)
        if len(binary) > 8_000_000:
            raise ValueError("Photo must be smaller than 8 MB")
        with Image.open(io.BytesIO(binary)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Use a JPEG, PNG or WebP photo")
            if image.width * image.height > 25_000_000:
                raise ValueError("Photo must have at most 25 million pixels")
            dimensions = {"width": image.width, "height": image.height, "format": image.format}
            image.verify()
        with Image.open(io.BytesIO(binary)) as image:
            from PIL import ImageStat
            image.thumbnail((512, 512))
            gray = image.convert("L")
            stats = ImageStat.Stat(gray)
            dimensions.update(mean_brightness=round(stats.mean[0], 2), brightness_scale="0–255", note="Image properties only. Fish identity, weight and age are not inferred.")
        result = analyze_observation(item.observation)
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    photo_id = uuid4().hex
    folder = ROOT / "data" / "photos"
    folder.mkdir(exist_ok=True)
    (folder / photo_id).write_bytes(binary)
    metadata = {"name": item.name, "observation": item.observation.model_dump(mode="json"), "result": result, "media_analysis": dimensions, "url": f"/api/photos/{photo_id}"}
    (folder / (photo_id + ".json")).write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


@router.get("/photos")
def photo_list():
    folder = ROOT / "data" / "photos"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]


@router.get("/photos/{photo_id}")
def photo_file(photo_id: str):
    import re
    from fastapi.responses import FileResponse
    if not re.fullmatch(r"[a-f0-9]{32}", photo_id):
        raise HTTPException(404)
    path = ROOT / "data" / "photos" / photo_id
    if not path.exists():
        raise HTTPException(404)
    from PIL import Image
    with Image.open(path) as image:
        media = Image.MIME[image.format]
    return FileResponse(path, media_type=media, headers={"X-Content-Type-Options": "nosniff"})


@router.post("/videos")
async def video_upload(request: Request, name: str = Query(max_length=200), observation: str = Query(max_length=4000)):
    """Bounded raw upload; decode actual pixels instead of trusting client metadata."""
    import math
    from uuid import uuid4
    try:
        item = Observation.model_validate_json(observation)
        result = analyze_observation(item)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        import cv2
    except ImportError as exc:
        raise HTTPException(503, "Video analysis needs opencv-python-headless installed in the project environment.") from exc
    if Path(name).suffix.lower() not in (".mp4", ".mov", ".webm", ".avi"):
        raise HTTPException(422, "Use MP4, MOV, WebM or AVI video")
    video_id = uuid4().hex
    folder = ROOT / "data" / "videos"
    folder.mkdir(exist_ok=True)
    path = folder / (video_id + Path(name).suffix.lower())
    generated = [path]
    capture = None
    try:
        total = 0
        with path.open("wb") as output:
            async for chunk in request.stream():
                total += len(chunk)
                if total > 50_000_000:
                    raise HTTPException(413, "Video must be at most 50 MB")
                output.write(chunk)
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not capture.isOpened() or not all(math.isfinite(x) and x > 0 for x in (fps, count, width, height)):
            raise HTTPException(422, "Video cannot be decoded. Try an MP4/H.264 export.")
        if width * height > 8_500_000 or count / fps > 180:
            raise HTTPException(422, "Use a clip up to 3 minutes and 8.5 million pixels per frame")
        frames = []
        for index, position in enumerate(np.linspace(0, max(0, int(count) - 1), min(5, int(count))).astype(int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok:
                continue
            brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            scale = min(1, 960 / frame.shape[1])
            frame = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))))
            frame_path = folder / f"{video_id}-{index}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise HTTPException(422, "Could not save extracted video frames")
            generated.append(frame_path)
            frames.append({"time_seconds": round(int(position) / fps, 2), "mean_brightness": round(brightness, 2), "url": f"/api/videos/{video_id}/frames/{index}"})
        if not frames:
            raise HTTPException(422, "Video contains no decodable frames")
        metadata = {"id": video_id, "name": name, "url": f"/api/videos/{video_id}/file", "observation": item.model_dump(mode="json"), "result": result,
                    "media_analysis": {"width": int(width), "height": int(height), "fps": round(fps, 3), "duration_seconds": round(count / fps, 2), "sampled_frames": frames,
                                       "note": "Frames and image properties were read. No trained fish detector or age/weight recognition model is available."}, "filename": path.name}
        (folder / (video_id + ".json")).write_text(json.dumps(metadata), encoding="utf-8")
        return metadata
    except BaseException:
        if capture is not None:
            capture.release()
        for created in generated:
            created.unlink(missing_ok=True)
        raise
    finally:
        if capture is not None:
            capture.release()


@router.get("/videos")
def video_list():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data" / "videos").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]


def video_metadata(video_id):
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", video_id):
        raise HTTPException(404)
    path = ROOT / "data" / "videos" / (video_id + ".json")
    if not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/videos/{video_id}/file")
def video_file(video_id: str):
    from fastapi.responses import FileResponse
    metadata = video_metadata(video_id)
    return FileResponse(ROOT / "data" / "videos" / metadata["filename"], headers={"X-Content-Type-Options": "nosniff"})


@router.get("/videos/{video_id}/frames/{index}")
def video_frame(video_id: str, index: int):
    from fastapi.responses import FileResponse
    video_metadata(video_id)
    if index not in range(5):
        raise HTTPException(404)
    path = ROOT / "data" / "videos" / f"{video_id}-{index}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


class CSVUpload(BaseModel):
    name: str = Field(max_length=200)
    content: str = Field(max_length=5_000_000)
    observation_date: date
    fish_count: int = Field(default=5000, ge=0, le=10000000)


def db():
    connection = sqlite3.connect(ROOT / "data" / "dashboard.sqlite3")
    connection.execute("CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY, name TEXT, payload TEXT)")
    return connection


@router.get("/uploads")
def uploads():
    with db() as conn:
        return [{"id": row[0], "name": row[1]} for row in conn.execute("SELECT id,name FROM uploads ORDER BY id DESC")]


@router.get("/uploads/{record_id}")
def upload_record(record_id: int):
    with db() as conn:
        row = conn.execute("SELECT payload FROM uploads WHERE id=?", (record_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Upload not found")
    return json.loads(row[0])


@router.post("/uploads")
def upload(item: CSVUpload):
    reader = csv.DictReader(io.StringIO(item.content.lstrip("\ufeff")))
    columns = reader.fieldnames or []
    if not columns or len(columns) != len(set(columns)):
        raise HTTPException(422, "CSV needs a unique header row")
    supported = {"avg_weight_g", "stocking_date", "hatch_date", "day", "temp_c", "ph", "do_mgl"}
    if not supported.intersection(columns):
        raise HTTPException(422, "No recognized columns. Download a CSV template for the supported headings.")
    rows, errors = [], []
    for line, raw in enumerate(reader, 2):
        if line > 10001:
            raise HTTPException(422, "Maximum 10,000 rows per upload")
        try:
            if None in raw:
                raise ValueError("Too many values for the header")
            values = {k: v.strip() for k, v in raw.items() if v and v.strip()}
            data = {k: values[k] for k in Observation.model_fields if k in values}
            data.setdefault("observation_date", item.observation_date)
            data.setdefault("fish_count", item.fish_count)
            obs = Observation(**data)
            result = analyze_observation(obs)
            sensor = {}
            for key in ("temp_c", "ph", "do_mgl"):
                if key in values:
                    number = float(values[key])
                    if not np.isfinite(number) or not ({"temp_c": (-5, 50), "ph": (0, 14), "do_mgl": (0, 50)}[key][0] <= number <= {"temp_c": (-5, 50), "ph": (0, 14), "do_mgl": (0, 50)}[key][1]):
                        raise ValueError(f"Invalid {key}")
                    sensor[key] = number
            if sensor and len(sensor) != 3:
                raise ValueError("Water readings require temp_c, ph and do_mgl together")
            status = STATUSES[int(rule_status(sensor["temp_c"], sensor["ph"], sensor["do_mgl"]))] if sensor else None
            feed = None
            if result["feeding"]["available"]:
                feed = result["feeding"]["daily_feed_kg"]
            rows.append({"row": line, "batch": values.get("batch", "Unassigned"), **obs.model_dump(mode="json"),
                         **sensor, **result, "status": status, "feed_kg": feed})
        except (ValueError, TypeError) as exc:
            errors.append({"row": line, "message": str(exc)[:400]})
    if not rows:
        raise HTTPException(422, {"message": "No valid data rows", "errors": errors[:50]})
    result = {"name": item.name, "rows": rows, "errors": errors[:50], "rejected_count": len(errors), "columns": columns}
    with db() as conn:
        cursor = conn.execute("INSERT INTO uploads(name,payload) VALUES (?,?)", (item.name, json.dumps(result, allow_nan=False)))
        result["id"] = cursor.lastrowid
    return result


@router.get("/metrics")
def metrics():
    path = MODELS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@router.get("/template/{kind}")
def template(kind: str):
    from fastapi.responses import Response
    templates = {
        "growth": "batch,observation_date,stocking_date,hatch_date,avg_weight_g,mean_temp_c,fish_count\nCage A,2026-10-17,2026-08-18,,110,29,5000\n",
        "sensor": "batch,observation_date,day,hour,temp_c,ph,do_mgl\nCage A,2026-10-17,60,15,30.2,8,6.4\n",
    }
    if kind not in templates:
        raise HTTPException(404)
    return Response(templates[kind], media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{kind}_template.csv"'})
