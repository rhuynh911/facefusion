import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy

from facefusion.face_analyser import get_many_faces
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_selector import sort_and_filter_faces
from facefusion.types import Face, VisionFrame
from facefusion.vision import count_video_frame_total, read_video_frame

CROP_SIZE = 256

ProgressCallback = Callable[[int, int, int], None]


@dataclass
class FaceCandidate:
	"""One distinct person found in a video, plus the frame that best identifies them."""
	embedding_sum : numpy.ndarray[Any, Any]
	appearance_count : int = 0
	first_frame_number : int = 0
	last_frame_number : int = 0
	reference_frame_number : int = 0
	reference_face_position : int = 0
	reference_quality : float = -1.0
	reference_crop : Optional[VisionFrame] = None
	max_face_height : float = 0.0
	gender : str = ''
	age : str = ''
	race : str = ''
	frame_numbers : List[int] = field(default_factory = list)

	def get_embedding_norm(self) -> numpy.ndarray[Any, Any]:
		norm = numpy.linalg.norm(self.embedding_sum)

		if norm > 0:
			return self.embedding_sum / norm
		return self.embedding_sum


def create_frame_numbers(target_path : str, video_fps : float, scan_every : float, scan_start : float, scan_end : Optional[float], max_samples : int) -> List[int]:
	frame_total = count_video_frame_total(target_path)
	first_frame_number = max(1, int(scan_start * video_fps))
	last_frame_number = min(frame_total, int(scan_end * video_fps)) if scan_end else frame_total
	frame_step = max(1, int(round(scan_every * video_fps)))
	frame_span = max(0, last_frame_number - first_frame_number)

	if frame_span // frame_step > max_samples:
		frame_step = max(1, frame_span // max_samples)

	return list(range(first_frame_number, last_frame_number + 1, frame_step))


def scan_video(target_path : str, frame_numbers : List[int], cluster_distance : float, progress_callback : Optional[ProgressCallback] = None) -> List[FaceCandidate]:
	candidates : List[FaceCandidate] = []

	for sample_index, frame_number in enumerate(frame_numbers):
		vision_frame = read_video_frame(target_path, frame_number)

		if vision_frame is not None:
			faces = sort_and_filter_faces(get_many_faces([ vision_frame ]))

			for face_position, face in enumerate(faces):
				candidate, identity_distance = find_candidate(candidates, face, cluster_distance)

				if not candidate:
					candidate = FaceCandidate(embedding_sum = numpy.zeros_like(face.embedding_norm), first_frame_number = frame_number)
					candidates.append(candidate)
					identity_distance = 0.0

				update_candidate(candidate, face, faces, vision_frame, frame_number, face_position, identity_distance)

		if progress_callback:
			progress_callback(sample_index + 1, len(frame_numbers), len(candidates))

	return merge_candidates(candidates, cluster_distance)


def find_candidate(candidates : List[FaceCandidate], face : Face, cluster_distance : float) -> Tuple[Optional[FaceCandidate], float]:
	best_candidate = None
	best_distance = 1.0

	for candidate in candidates:
		identity_distance = calculate_identity_distance(face.embedding_norm, candidate.get_embedding_norm())

		if identity_distance < best_distance:
			best_distance = identity_distance
			best_candidate = candidate

	if best_candidate and best_distance < cluster_distance:
		return best_candidate, best_distance
	return None, best_distance


def update_candidate(candidate : FaceCandidate, face : Face, faces : List[Face], vision_frame : VisionFrame, frame_number : int, face_position : int, identity_distance : float) -> None:
	start_x, start_y, end_x, end_y = face.bounding_box
	# max dimension, because a face detected at 90 degrees has its width and height swapped
	face_height = min(1.0, max(float(end_y - start_y), float(end_x - start_x)) / vision_frame.shape[0])
	candidate.embedding_sum = candidate.embedding_sum + face.embedding_norm
	candidate.appearance_count += 1
	candidate.last_frame_number = frame_number
	candidate.frame_numbers.append(frame_number)
	candidate.max_face_height = max(candidate.max_face_height, face_height)
	reference_quality = calculate_reference_quality(face, faces, vision_frame, identity_distance)

	if reference_quality > candidate.reference_quality:
		candidate.reference_quality = reference_quality
		candidate.reference_frame_number = frame_number
		candidate.reference_face_position = face_position
		candidate.reference_crop = create_face_crop(vision_frame, face)
		candidate.gender = str(face.gender)
		candidate.age = format_age(face.age)
		candidate.race = str(face.race)


def merge_candidates(candidates : List[FaceCandidate], cluster_distance : float) -> List[FaceCandidate]:
	"""Second pass, because a drifting centroid can split one person into two clusters during the scan."""
	merged_candidates : List[FaceCandidate] = []

	for candidate in sorted(candidates, key = lambda item: item.appearance_count, reverse = True):
		target_candidate = None

		for merged_candidate in merged_candidates:
			if calculate_identity_distance(candidate.get_embedding_norm(), merged_candidate.get_embedding_norm()) < cluster_distance:
				target_candidate = merged_candidate
				break

		if target_candidate:
			target_candidate.embedding_sum = target_candidate.embedding_sum + candidate.embedding_sum
			target_candidate.appearance_count += candidate.appearance_count
			target_candidate.first_frame_number = min(target_candidate.first_frame_number, candidate.first_frame_number)
			target_candidate.last_frame_number = max(target_candidate.last_frame_number, candidate.last_frame_number)
			target_candidate.frame_numbers.extend(candidate.frame_numbers)
			target_candidate.max_face_height = max(target_candidate.max_face_height, candidate.max_face_height)

			if candidate.reference_quality > target_candidate.reference_quality:
				target_candidate.reference_quality = candidate.reference_quality
				target_candidate.reference_frame_number = candidate.reference_frame_number
				target_candidate.reference_face_position = candidate.reference_face_position
				target_candidate.reference_crop = candidate.reference_crop
				target_candidate.gender = candidate.gender
				target_candidate.age = candidate.age
				target_candidate.race = candidate.race
		else:
			merged_candidates.append(candidate)

	return sorted(merged_candidates, key = lambda item: item.appearance_count, reverse = True)


def filter_candidates(candidates : List[FaceCandidate], min_appearances : int, min_face_height : float, max_faces : int) -> Tuple[List[FaceCandidate], List[str]]:
	"""Returns the candidates worth showing plus notes about everything that was held back."""
	notes : List[str] = []
	filtered_candidates = [ candidate for candidate in candidates if candidate.appearance_count >= min_appearances and candidate.max_face_height >= min_face_height ]

	if not filtered_candidates:
		notes.append('every face was filtered out, showing all of them instead')
		filtered_candidates = candidates
	else:
		dropped_total = len(candidates) - len(filtered_candidates)

		if dropped_total:
			notes.append('hid {} background/one-off face(s), lower the minimum face size to see them'.format(dropped_total))

	if len(filtered_candidates) > max_faces:
		notes.append('showing the {} most frequent of {} faces'.format(max_faces, len(filtered_candidates)))
		filtered_candidates = filtered_candidates[:max_faces]

	return filtered_candidates, notes


def find_similar_candidates(candidates : List[FaceCandidate], candidate_index : int, reference_face_distance : float) -> List[Tuple[int, float]]:
	"""Other people close enough that facefusion would swap them too."""
	similar_candidates = []
	selected_candidate = candidates[candidate_index]

	for other_index, other_candidate in enumerate(candidates):
		if other_index != candidate_index:
			identity_distance = calculate_identity_distance(selected_candidate.get_embedding_norm(), other_candidate.get_embedding_norm())

			if identity_distance < reference_face_distance:
				similar_candidates.append((other_index, identity_distance))

	return similar_candidates


def calculate_identity_distance(embedding_norm : numpy.ndarray[Any, Any], reference_embedding_norm : numpy.ndarray[Any, Any]) -> float:
	face_distance = 1 - numpy.dot(embedding_norm, reference_embedding_norm)
	return float(numpy.interp(face_distance, [ 0, 2 ], [ 0, 1 ]))


def calculate_reference_quality(face : Face, faces : List[Face], vision_frame : VisionFrame, identity_distance : float) -> float:
	frame_height, frame_width = vision_frame.shape[:2]
	area_ratio = calculate_face_area(face) / float(frame_height * frame_width)
	size_score = min(1.0, math.sqrt(max(area_ratio, 0.0) / 0.02))
	detector_score = float(face.score_set.get('detector') or 0.0)
	frontal_score = calculate_frontal_score(face)
	identity_score = max(0.0, 1.0 - identity_distance * 2)
	ambiguity_score = calculate_ambiguity_score(face, faces)
	return detector_score * size_score * frontal_score * ambiguity_score * (0.5 + 0.5 * identity_score)


def calculate_frontal_score(face : Face) -> float:
	face_landmark_5 = face.landmark_set.get('5/68')

	if face_landmark_5 is None or len(face_landmark_5) < 3:
		return 1.0

	left_eye, right_eye, nose = face_landmark_5[0], face_landmark_5[1], face_landmark_5[2]
	eye_distance = float(numpy.linalg.norm(right_eye - left_eye))

	if eye_distance <= 0:
		return 1.0

	eye_center_x = (left_eye[0] + right_eye[0]) / 2
	nose_offset = abs(float(nose[0]) - float(eye_center_x)) / eye_distance
	return float(max(0.0, 1.0 - nose_offset * 1.5))


def calculate_ambiguity_score(face : Face, faces : List[Face]) -> float:
	"""Prefer reference frames where the chosen face cannot be confused with a neighbour of near-identical size."""
	if len(faces) < 2:
		return 1.0

	face_area = calculate_face_area(face)

	for other_face in faces:
		if other_face is not face:
			other_area = calculate_face_area(other_face)

			if face_area > 0 and abs(other_area - face_area) / face_area < 0.1:
				return 0.5

	return 0.9


def calculate_face_area(face : Face) -> float:
	start_x, start_y, end_x, end_y = face.bounding_box
	return max(0.0, float(end_x - start_x)) * max(0.0, float(end_y - start_y))


def create_face_crop(vision_frame : VisionFrame, face : Face) -> VisionFrame:
	"""Warp to the ffhq template so tilted and sideways detections still show up upright."""
	face_landmark_5 = face.landmark_set.get('5/68')

	if face_landmark_5 is not None:
		crop_vision_frame, _ = warp_face_by_face_landmark_5(vision_frame, face_landmark_5, 'ffhq_512', (CROP_SIZE, CROP_SIZE))
		return crop_vision_frame
	return numpy.zeros((CROP_SIZE, CROP_SIZE, 3), dtype = numpy.uint8)


def format_age(age : Any) -> str:
	if isinstance(age, range):
		return str(age.start) + '-' + str(age.stop)
	return str(age)


def format_timestamp(frame_number : int, video_fps : float) -> str:
	total_seconds = int(frame_number / video_fps) if video_fps else 0
	return '{:02d}:{:02d}'.format(total_seconds // 60, total_seconds % 60)
