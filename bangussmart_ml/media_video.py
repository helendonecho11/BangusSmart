"""Sample, measure and associate candidates; track count is always an estimate."""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from media_estimation import length_to_weight, summarize_fish
from media_vision import MeasurementError, load_config, measure_image


class FishTracker:
    """ASSUMPTION: position/size assignment, no appearance-trained identity model.

    Coordinates are cm relative to the visible board. Occlusion, crossings,
    rapid movement and re-entry can create duplicate or merged identities.
    """
    def __init__(self, config):
        self.config = config['video']
        self.tracks = []

    def update(self, fish, reference, time_seconds):
        origin = np.asarray(reference['center_px'])
        basis = np.array([reference['x_axis'], reference['y_axis']])
        centers = [(basis @ (np.asarray(f['centroid_px']) - origin)) / reference['pixels_per_cm'] for f in fish]
        active = [t for t in self.tracks if time_seconds - t['last_time'] <= self.config['max_tracking_gap_seconds']]
        costs = np.full((len(active), len(fish)), 1e6)
        for i, track in enumerate(active):
            prior_length = float(np.median(track['lengths']))
            for j, candidate in enumerate(fish):
                change = abs(candidate['length_cm'] - prior_length) / prior_length
                distance = np.linalg.norm(centers[j] - track['last_center']) / prior_length
                if change <= self.config['max_length_change_fraction'] and distance <= self.config['max_centroid_move_lengths']:
                    costs[i, j] = distance + change
        assigned = {}
        if costs.size:
            track_indices, detection_indices = linear_sum_assignment(costs)
            assigned = {int(j): active[int(i)] for i, j in zip(track_indices, detection_indices) if costs[i, j] < 1e6}
        for index, candidate in enumerate(fish):
            track = assigned.get(index)
            if track is None:
                track = {'track_id': len(self.tracks) + 1, 'lengths': [], 'weights': [], 'first_time': time_seconds}
                self.tracks.append(track)
            track.update(last_time=time_seconds, last_center=centers[index])
            track['lengths'].append(candidate['length_cm'])
            track['weights'].append(candidate['estimated_weight']['point_g'])
            candidate['track_id'] = track['track_id']
        return fish


def analyze_video(path, options, output_dir, output_url, config=None):
    config = config or load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not capture.isOpened() or not all(np.isfinite(v) and v > 0 for v in (fps, count, width, height)):
            raise MeasurementError('invalid_video', 'Video cannot be decoded. Try a short MP4/H.264 export.')
        duration = count / fps
        if duration > config['video']['max_duration_seconds'] or width * height > 8500000:
            raise MeasurementError('video_limits', 'Use a clip up to 180 seconds and 8.5 million pixels per frame.')
        sampling = min(options.sample_fps, config['video']['max_sampled_frames'] / duration)
        positions = np.unique(np.floor(np.arange(0, duration, 1 / sampling) * fps).astype(int))
        positions = positions[positions < int(count)][:config['video']['max_sampled_frames']]
        tracker = FishTracker(config)
        quality, annotations = [], []
        for frame_index in positions:
            time = float(frame_index / fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = capture.read()
            if not ok:
                quality.append({'time_seconds': time, 'accepted': False, 'code': 'decode_failure'})
                continue
            try:
                measurement, annotated = measure_image(image, options.square_size_cm, config)
                for fish in measurement['fish']:
                    fish['estimated_weight'] = length_to_weight(fish['length_cm'], config)
                tracker.update(measurement['fish'], measurement['reference'], time)
                for fish in measurement['fish']:
                    x, y, _, _ = fish['bbox_px']
                    cv2.putText(annotated, f"Track {fish['track_id']}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 70, 20), 2)
                quality.append({'time_seconds': time, 'accepted': True, 'fish_candidates': len(measurement['fish']),
                                'quality': measurement['quality'], 'pixels_per_cm': measurement['reference']['pixels_per_cm']})
                if len(annotations) < 6:
                    filename = f'frame_{len(annotations)}.jpg'
                    if not cv2.imwrite(str(output_dir / filename), annotated):
                        raise OSError('Could not write annotated video frame')
                    annotations.append({'time_seconds': time, 'path': f'{output_url}/{filename}'})
            except MeasurementError as exc:
                quality.append({'time_seconds': time, 'accepted': False, 'code': exc.code, 'message': str(exc), 'quality': exc.quality})
        accepted = sum(q['accepted'] for q in quality)
        if not accepted:
            raise MeasurementError('no_usable_video_frames', 'All sampled frames were rejected. Include a sharp side-view fish and the checkerboard in every frame.', {'frames': quality, 'sampled_count': len(quality), 'rejected_fraction': 1.0})
        tracks = []
        for track in tracker.tracks:
            weights = track['weights']
            median_weight = float(np.median(weights)) if weights and all(w is not None for w in weights) else None
            tracks.append({'fish_id': track['track_id'], 'track_id': track['track_id'],
                           'length_cm': float(np.median(track['lengths'])), 'length_type': 'fork_length_proxy',
                           'samples': len(track['lengths']), 'first_seen_seconds': track['first_time'], 'last_seen_seconds': track['last_time'],
                           'confidence': 'low', 'estimated_weight': {'point_g': median_weight, 'confidence': 'low' if median_weight is not None else 'unavailable',
                                                                   'source': 'median_of_per_frame_length_weight_estimates',
                                                                   'warnings': ['Track identity and length-weight conversion are unvalidated.']}})
        result = summarize_fish(tracks, options, config, video_count=len(tracks))
        result.update(media_type='video', estimated_distinct_fish_seen=len(tracks), annotated_output=annotations,
                      video={'duration_seconds': duration, 'source_fps': fps, 'requested_sample_fps': options.sample_fps,
                             'effective_sample_fps': sampling, 'sampled_count': len(quality), 'accepted_count': accepted,
                             'rejected_fraction': 1 - accepted / len(quality), 'frames': quality})
        result['warnings'].extend(['Distinct tracks are an estimated number of fish seen, not a verified batch count. User fish count takes priority.',
                                   'Keep the camera and board stationary; occlusion, crossings, exits/re-entry and skipped frames can split or merge tracks.'])
        if sampling < options.sample_fps:
            result['warnings'].append('Sample rate reduced to respect the configured maximum frame count.')
        return result
    finally:
        capture.release()
