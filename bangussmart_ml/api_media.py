"""Standalone: python -m uvicorn api_media:app --host 127.0.0.1 --port 8001.

The same router is included in api:app. Upload raw file bytes and pass the
AnalysisOptions JSON in the `options` query parameter; no multipart dependency.
"""
import json
import re
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from media_estimation import AnalysisOptions, summarize_fish
from media_vision import ROOT, MeasurementError, decode_image, load_config, measure_image
from media_video import analyze_video
from media_csv import analyze_csv

router = APIRouter()
OUTPUT_ROOT = ROOT / 'data' / 'media_results'
STATIC = ROOT / 'static'


def parse_options(options):
    try:
        return AnalysisOptions.model_validate_json(options)
    except ValueError as exc:
        raise HTTPException(422, {'code': 'invalid_options', 'message': str(exc)}) from exc


async def read_upload(request, limit):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, {'code': 'upload_too_large', 'message': f'Upload limit is {limit} bytes'})
        data.extend(chunk)
    if not data:
        raise HTTPException(422, {'code': 'empty_upload', 'message': 'Choose a nonempty file.'})
    return bytes(data)


def save_result(result, folder, run_id, config):
    # Persist the actual configuration so a later refit cannot rewrite history.
    result.update(run_id=run_id, configuration=config, real_fish_accuracy='not_evaluated_no_real_labels',
                  decision_support_note='Assists farm personnel; does not replace their judgment.')
    result['result_path'] = f'/media-results/{run_id}/result.json'
    (folder / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def measurement_error(exc):
    return HTTPException(422, {'code': exc.code, 'message': str(exc), 'quality': exc.quality,
                              'age': {'point_days': None, 'range_days': None, 'confidence': 'unavailable'},
                              'feed_kg': None})


@router.get('/media')
def upload_page():
    return FileResponse(STATIC / 'media.html')


@router.get('/media/reference.svg')
def reference_board():
    from fastapi.responses import Response
    # Print at 100%; measure physical squares before supplying square_size_cm.
    squares = ''.join(f'<rect x="{10+c*10}" y="{10+r*10}" width="10" height="10" fill="{"#000" if (r+c)%2==0 else "#fff"}"/>' for r in range(6) for c in range(8))
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="90mm" viewBox="0 0 100 90"><rect width="100" height="90" fill="white"/>'+squares+'<text x="5" y="80" font-size="3">8 x 6 squares; measure printed square size.</text><text x="5" y="85" font-size="3">Intended 1 cm squares. Print at 100%.</text></svg>'
    return Response(svg, media_type='image/svg+xml', headers={'Content-Disposition': 'attachment; filename="reference_checkerboard.svg"'})


@router.get('/media-results/{run_id}/{filename}')
def artifact(run_id: str, filename: str):
    if not re.fullmatch(r'[a-f0-9]{32}', run_id) or not re.fullmatch(r'(annotated\.png|frame_[0-5]\.jpg|result\.json)', filename):
        raise HTTPException(404, 'Artifact not found')
    path = OUTPUT_ROOT / run_id / filename
    if not path.is_file():
        raise HTTPException(404, 'Artifact not found')
    return FileResponse(path, headers={'X-Content-Type-Options': 'nosniff'})


RAW_BODY = {'requestBody': {'required': True, 'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}}}}


@router.post('/analyze/image', openapi_extra=RAW_BODY)
async def image_analysis(request: Request, options: str = Query(max_length=5000)):
    settings = parse_options(options)
    config = load_config()
    binary = await read_upload(request, 8_000_000)
    try:
        image = decode_image(binary, config)
        measurement, annotated = measure_image(image, settings.square_size_cm, config)
        result = summarize_fish(measurement['fish'], settings, config)
    except MeasurementError as exc:
        raise measurement_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Missing trained growth model. Run python bangussmart_ml.py all.') from exc
    run_id = uuid4().hex
    folder = OUTPUT_ROOT / run_id
    folder.mkdir(parents=True)
    if not cv2.imwrite(str(folder / 'annotated.png'), annotated):
        raise HTTPException(500, 'Could not save annotated image')
    result.update(media_type='image', detected_fish_count=measurement['detected_fish_count'], reference=measurement['reference'], detector_used=measurement['detector_used'],
                  quality=measurement['quality'], annotated_output=[{'path': f'/media-results/{run_id}/annotated.png'}])
    result['warnings'].extend(measurement['warnings'])
    return save_result(result, folder, run_id, config)


@router.post('/analyze/video', openapi_extra=RAW_BODY)
async def video_analysis(request: Request, options: str = Query(max_length=5000), filename: str = Query(default='clip.mp4', max_length=200)):
    settings = parse_options(options)
    config = load_config()
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.mp4', '.mov', '.webm', '.avi'):
        raise HTTPException(422, 'Use MP4, MOV, WebM or AVI video.')
    run_id = uuid4().hex
    folder = OUTPUT_ROOT / run_id
    folder.mkdir(parents=True)
    path = folder / ('input' + suffix)
    try:
        total = 0
        with path.open('wb') as output:
            async for chunk in request.stream():
                total += len(chunk)
                if total > config['video']['max_upload_bytes']:
                    raise HTTPException(413, 'Video upload limit is 50 MB')
                output.write(chunk)
        result = analyze_video(path, settings, folder, f'/media-results/{run_id}', config)
        return save_result(result, folder, run_id, config)
    except MeasurementError as exc:
        raise measurement_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Missing trained growth model. Train the prototype first.') from exc
    finally:
        path.unlink(missing_ok=True)


@router.post('/analyze/csv', openapi_extra=RAW_BODY)
async def csv_analysis(request: Request, options: str = Query(default='{"square_size_cm":1}', max_length=5000)):
    settings = parse_options(options)
    config = load_config()
    binary = await read_upload(request, 5_000_000)
    try:
        content = binary.decode('utf-8-sig')
        result = analyze_csv(content, settings, config)
    except UnicodeError as exc:
        raise HTTPException(422, 'CSV must be UTF-8 encoded.') from exc
    except MeasurementError as exc:
        raise measurement_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, 'Missing trained growth model. Train the prototype first.') from exc
    run_id = uuid4().hex
    folder = OUTPUT_ROOT / run_id
    folder.mkdir(parents=True)
    return save_result(result, folder, run_id, config)


app = FastAPI(title='BangusSmart media measurement prototype', description='Protocol-constrained geometry and uncertain culture-age estimates; not species identification or exact age recognition.')
app.include_router(router)
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def home():
    return upload_page()
