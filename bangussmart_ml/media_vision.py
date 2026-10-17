"""Classical, protocol-constrained geometry. Does not establish species or exact age."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _local_yolo(path):
    # This function is reached only after a local file exists. Never accept a
    # model name/URL that Ultralytics could interpret as an instruction to download.
    from ultralytics import YOLO
    return YOLO(str(path), task='segment')


def segmentation_mask(image, gray, board_mask, threshold, config):
    fallback = np.where((gray < threshold) & (board_mask == 0), 255, 0).astype(np.uint8)
    detector = config.get('detector', {})
    if detector.get('backend', 'classical') == 'classical':
        return fallback, 'classical_contours', []
    weights = detector.get('local_yolo_weights')
    path = Path(weights) if weights else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    if detector.get('backend') != 'local_yolo' or path is None or not path.is_file():
        return fallback, 'classical_contours', ['Requested local segmentation weights are absent; used classical fallback. No weights downloaded.']
    try:
        prediction = _local_yolo(path.resolve()).predict(source=image, device='cpu', verbose=False)[0]
        masks = prediction.masks
        if masks is None:
            return fallback, 'classical_contours', ['Local segmentation model returned no masks; used classical fallback.']
        output = np.zeros(gray.shape, np.uint8)
        allowed = set(detector['fish_class_names'])
        for polygon, class_id in zip(masks.xy, prediction.boxes.cls.cpu().numpy()):
            if str(prediction.names[int(class_id)]).lower() in allowed:
                cv2.fillPoly(output, [np.rint(polygon).astype(np.int32)], 255)
        output[board_mask != 0] = 0
        if output.any():
            return output, 'local_yolo_segmentation', ['Local model species labels are not independently validated; review candidates.']
        return fallback, 'classical_contours', ['Local model has no detected configured fish classes; used classical fallback.']
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        return fallback, 'classical_contours', [f'Optional local model unavailable ({type(exc).__name__}); used classical fallback.']


class MeasurementError(ValueError):
    def __init__(self, code, message, quality=None):
        super().__init__(message)
        self.code = code
        self.quality = quality or {}


def load_config(path=None):
    return json.loads(Path(path or ROOT / 'media_config.json').read_text(encoding='utf-8'))


def decode_image(binary, config=None):
    from PIL import Image, UnidentifiedImageError
    import io
    config = config or load_config()
    try:
        with Image.open(io.BytesIO(binary)) as image:
            if min(image.width, image.height) < 100:
                raise MeasurementError('image_too_small', 'Image is too small to measure a checkerboard and fish. Use at least 100 pixels on each side.')
            if image.width * image.height > config['measurement']['max_image_pixels']:
                raise MeasurementError('image_too_large', 'Image exceeds 25 million pixels.')
            if image.format not in ('JPEG', 'PNG', 'WEBP'):
                raise MeasurementError('unsupported_image', 'Use a JPEG, PNG or WebP image.')
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise MeasurementError('invalid_image', 'Image could not be decoded.') from exc
    decoded = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise MeasurementError('invalid_image', 'Image contains no decodable pixels.')
    return decoded


def detect_reference(image, square_size_cm, config):
    if not np.isfinite(square_size_cm) or square_size_cm <= 0:
        raise MeasurementError('invalid_reference', 'Enter the measured checkerboard square size in cm.')
    pattern = tuple(config['reference']['inner_corners'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Downsize only for corner detection; return all coordinates in original pixels.
    ratio = min(1., 1600 / max(gray.shape))
    resized = cv2.resize(gray, None, fx=ratio, fy=ratio) if ratio < 1 else gray
    found, corners = cv2.findChessboardCornersSB(resized, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise MeasurementError('cannot_measure_scale', 'Cannot measure scale: no 8 × 6 square checkerboard detected. Include a flat board of known square size in the fish plane; retake the side-view photo.')
    grid = corners.reshape(pattern[1], pattern[0], 2).astype(float) / ratio
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()
    distances = np.concatenate([horizontal, vertical])
    spacing_ratio = float(np.percentile(distances, 95) / np.percentile(distances, 5))
    hvec = np.mean(np.diff(grid, axis=1), axis=(0, 1))
    vvec = np.mean(np.diff(grid, axis=0), axis=(0, 1))
    skew = abs(float(np.dot(hvec, vvec) / (np.linalg.norm(hvec) * np.linalg.norm(vvec))))
    if spacing_ratio > config['reference']['max_spacing_ratio'] or skew > .15:
        raise MeasurementError('perspective_reference', 'Reference is distorted or tilted. Hold the camera perpendicular to the fish and checkerboard.', {'spacing_ratio': spacing_ratio, 'axis_cosine': skew})
    # Exclude the complete board, including its outside squares, from segmentation.
    edge = float(np.median(distances))
    hull = cv2.convexHull(grid.astype(np.float32).reshape(-1, 1, 2))
    board_mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(board_mask, hull.astype(np.int32), 255)
    margin = max(3, int(np.ceil(edge * 1.5)))
    board_mask = cv2.dilate(board_mask, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8))
    return {'pixels_per_cm': edge / square_size_cm, 'square_size_cm': float(square_size_cm),
            'spacing_ratio': spacing_ratio, 'inner_corners': list(pattern), 'method': 'detected_checkerboard',
            'center_px': grid.mean(axis=(0, 1)).tolist(),
            'x_axis': (hvec / np.linalg.norm(hvec)).tolist(), 'y_axis': (vvec / np.linalg.norm(vvec)).tolist(),
            'physical_size_source': 'user_supplied_not_independently_verified'}, board_mask, grid


def contour_measurement(contour, ppcm, shape, config):
    points = contour[:, 0, :].astype(float)
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis, normal = vectors[0], vectors[1]
    along, across = (points - center) @ axis, (points - center) @ normal
    lo, hi = float(along.min()), float(along.max())
    extent = hi - lo
    width = float(across.max() - across.min())
    aspect = extent / max(width, 1)
    if not config['measurement']['min_aspect_ratio'] <= aspect <= config['measurement']['max_aspect_ratio']:
        raise MeasurementError('not_fish_shaped', 'Contour is not an isolated elongated fish candidate.')
    x, y, w, h = cv2.boundingRect(contour)
    if x <= 2 or y <= 2 or x + w >= shape[1] - 2 or y + h >= shape[0] - 2:
        raise MeasurementError('clipped_fish', 'A fish touches the image edge; show the complete fish.')
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull_indices) if len(hull_indices) >= 3 else None
    except cv2.error as exc:
        raise MeasurementError('ambiguous_outline', 'Contour is self-intersecting or ambiguous; use a clearer, isolated fish outline.') from exc
    fork_options = []
    mid_across = (across.min() + across.max()) / 2
    if defects is not None:
        for _, _, far_index, fixed_depth in defects.reshape(-1, 4):
            longitudinal, lateral = along[far_index], across[far_index]
            near_end = min(longitudinal - lo, hi - longitudinal) < .14 * extent
            if near_end and abs(lateral - mid_across) < .22 * width and fixed_depth / 256 > max(2, .025 * extent):
                fork_options.append((fixed_depth, int(far_index)))
    if not fork_options:
        raise MeasurementError('fork_not_visible', 'Cannot locate a tail-fork landmark. Use an isolated, straight side-view fish with the complete tail visible.')
    _, fork_index = max(fork_options)
    fork = points[fork_index]
    fork_projection = along[fork_index]
    snout_index = int(np.argmin(along) if fork_projection > (lo + hi) / 2 else np.argmax(along))
    snout = points[snout_index]
    # Projected snout-to-fork, NOT total tail-tip length. Heuristic landmark; review annotation.
    length_px = abs(float((fork - snout) @ axis))
    return {'length_cm': length_px / ppcm, 'length_type': 'fork_length_proxy',
            'snout_px': snout.tolist(), 'fork_px': fork.tolist(), 'centroid_px': center.tolist(),
            'bbox_px': [x, y, w, h], 'aspect_ratio': aspect,
            'confidence': 'low', 'landmark_verified': False}


def measure_image(image, square_size_cm, config=None):
    config = config or load_config()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference, board_mask, grid = detect_reference(image, square_size_cm, config)
    # Local sharpness excludes the checkerboard, which otherwise hides blurry fish.
    background = gray[board_mask == 0]
    if not len(background):
        raise MeasurementError('no_fish', 'Reference fills the view; include the fish beside it.')
    threshold = min(210, float(np.percentile(background, 85)) - 30)
    mask, detector_used, detector_warnings = segmentation_mask(image, gray, board_mask, threshold, config)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= config['measurement']['min_contour_area_px']]
    if len(contours) > config['measurement']['max_fish']:
        raise MeasurementError('cluttered_image', 'Too many objects. Use a plain tray and separated fish.')
    fish, rejected = [], []
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, tuple(reference['inner_corners']), grid.astype(np.float32).reshape(-1, 1, 2), True)
    sharpness_values = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[max(0, y - 5):y + h + 5, max(0, x - 5):x + w + 5]
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        sharpness_values.append(sharpness)
        try:
            if sharpness < config['measurement']['min_blur_variance']:
                raise MeasurementError('blurry_fish', 'Fish outline is blurry. Retake with even lighting and a steady camera.')
            candidate = contour_measurement(contour, reference['pixels_per_cm'], image.shape, config)
            candidate['fish_id'] = len(fish) + 1
            candidate['sharpness_variance'] = sharpness
            fish.append(candidate)
            cv2.drawContours(annotated, [contour], -1, (70, 155, 30), 2)
            snout, fork = tuple(np.rint(candidate['snout_px']).astype(int)), tuple(np.rint(candidate['fork_px']).astype(int))
            cv2.line(annotated, snout, fork, (40, 90, 230), 2)
            for endpoint in (snout, fork):
                cv2.circle(annotated, endpoint, 5, (40, 90, 230), -1)
            cv2.putText(annotated, f"#{candidate['fish_id']} {candidate['length_cm']:.2f} cm FL proxy", (x, max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 90, 35), 2)
        except MeasurementError as exc:
            rejected.append({'code': exc.code, 'message': str(exc), 'bbox_px': [x, y, w, h]})
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (20, 100, 230), 2)
    if not fish:
        code = 'blurry_image' if rejected and all(r['code'] == 'blurry_fish' for r in rejected) else 'no_measurable_fish'
        raise MeasurementError(code, 'No measurable fish candidates. Use separated, straight fish on a light background with visible tails.', {'rejected_candidates': rejected, 'sharpness_variance': sharpness_values})
    cv2.putText(annotated, 'LOW confidence: review snout/fork marks. Not species or age recognition.', (15, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, .50, (25, 45, 180), 1)
    return {'fish': fish, 'detected_fish_count': len(fish), 'reference': reference, 'detector_used': detector_used,
            'quality': {'rejected_candidates': rejected, 'sharpness_variance': sharpness_values},
            'confidence': 'low', 'warnings': detector_warnings + ['Contours are unverified fish candidates, not confirmed species detection.', 'Fork landmarks are heuristic: review the annotated image.', 'Scale is valid only when the operator supplies the true square size and board and fish share a plane.']}, annotated
