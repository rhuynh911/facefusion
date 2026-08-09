from typing import Any, List, Optional, Tuple

import cv2
import gradio
import numpy
from gradio_rangeslider import RangeSlider

import facefusion.choices
from facefusion import state_manager, translator
from facefusion.audio import create_empty_audio_frame, get_voice_frame
from facefusion.common_helper import calculate_float_step, calculate_int_step, get_first
from facefusion.face_analyser import get_many_faces
from facefusion.face_scanner import FaceCandidate, create_face_crop, create_frame_numbers, filter_candidates, find_similar_candidates, format_age, format_timestamp, scan_video
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import filter_audio_paths, is_image, is_video
from facefusion.types import FaceSelectorOrder, Gender, Race
from facefusion.uis import choices as uis_choices
from facefusion.uis.core import get_ui_components
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import detect_video_fps, read_static_image, read_static_images, read_video_frame

FACE_SCANNER_SCAN_BUTTON : Optional[gradio.Button] = None
FACE_SCANNER_GALLERY : Optional[gradio.Gallery] = None
FACE_SCANNER_STATUS : Optional[gradio.Markdown] = None
FACE_SCANNER_TEST_BUTTON : Optional[gradio.Button] = None
FACE_SCANNER_TEST_IMAGE : Optional[gradio.Image] = None
FACE_SCANNER_SCAN_EVERY_SLIDER : Optional[gradio.Slider] = None
FACE_SCANNER_MIN_FACE_SIZE_SLIDER : Optional[gradio.Slider] = None
FACE_SCANNER_MAX_FACES_SLIDER : Optional[gradio.Slider] = None
FACE_SCANNER_ORDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SCANNER_GENDER_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SCANNER_RACE_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SCANNER_AGE_RANGE_SLIDER : Optional[RangeSlider] = None
FACE_SCANNER_REFERENCE_FACE_DISTANCE_SLIDER : Optional[gradio.Slider] = None

FACE_CANDIDATES : List[FaceCandidate] = []
SELECTED_CANDIDATE_INDEX : Optional[int] = None
VIDEO_FPS = 25.0
# grouping is deliberately not exposed, 0.3 is the value the scanner was tuned against
CLUSTER_DISTANCE = 0.3
MAX_SAMPLES = 400
IDLE_STATUS = 'Load a target, then press **SCAN FOR FACES**.'


def render() -> None:
	global FACE_SCANNER_SCAN_BUTTON
	global FACE_SCANNER_GALLERY
	global FACE_SCANNER_STATUS
	global FACE_SCANNER_TEST_BUTTON
	global FACE_SCANNER_TEST_IMAGE
	global FACE_SCANNER_SCAN_EVERY_SLIDER
	global FACE_SCANNER_MIN_FACE_SIZE_SLIDER
	global FACE_SCANNER_MAX_FACES_SLIDER
	global FACE_SCANNER_ORDER_DROPDOWN
	global FACE_SCANNER_GENDER_DROPDOWN
	global FACE_SCANNER_RACE_DROPDOWN
	global FACE_SCANNER_AGE_RANGE_SLIDER
	global FACE_SCANNER_REFERENCE_FACE_DISTANCE_SLIDER

	FACE_SCANNER_SCAN_BUTTON = gradio.Button(
		value = 'SCAN FOR FACES',
		variant = 'primary',
		size = 'sm'
	)
	FACE_SCANNER_GALLERY = gradio.Gallery(
		label = 'FACES IN THE TARGET',
		object_fit = 'cover',
		columns = 6,
		allow_preview = False,
		show_label = True,
		elem_classes = 'box-face-scanner'
	)
	FACE_SCANNER_STATUS = gradio.Markdown(value = IDLE_STATUS)
	with gradio.Row():
		FACE_SCANNER_TEST_BUTTON = gradio.Button(
			value = 'TEST THIS FACE',
			size = 'sm'
		)
	FACE_SCANNER_TEST_IMAGE = gradio.Image(
		label = 'TEST FRAME (BEFORE / AFTER)',
		visible = False
	)
	with gradio.Group():
		with gradio.Row():
			FACE_SCANNER_SCAN_EVERY_SLIDER = gradio.Slider(
				label = 'SCAN EVERY (SECONDS)',
				value = 1.0,
				minimum = 0.25,
				maximum = 5.0,
				step = 0.25
			)
			FACE_SCANNER_MIN_FACE_SIZE_SLIDER = gradio.Slider(
				label = 'MINIMUM FACE SIZE (% OF FRAME)',
				value = 5,
				minimum = 0,
				maximum = 40,
				step = 1
			)
			FACE_SCANNER_MAX_FACES_SLIDER = gradio.Slider(
				label = 'MAXIMUM FACES SHOWN',
				value = 24,
				minimum = 4,
				maximum = 48,
				step = 1
			)
	with gradio.Group():
		with gradio.Row():
			FACE_SCANNER_ORDER_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_order_dropdown'),
				choices = facefusion.choices.face_selector_orders,
				value = state_manager.get_item('face_selector_order')
			)
			FACE_SCANNER_GENDER_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_gender_dropdown'),
				choices = [ 'none' ] + facefusion.choices.face_selector_genders,
				value = state_manager.get_item('face_selector_gender') or 'none'
			)
			FACE_SCANNER_RACE_DROPDOWN = gradio.Dropdown(
				label = translator.get('uis.face_selector_race_dropdown'),
				choices = [ 'none' ] + facefusion.choices.face_selector_races,
				value = state_manager.get_item('face_selector_race') or 'none'
			)
		with gradio.Row():
			face_selector_age_start = state_manager.get_item('face_selector_age_start') or facefusion.choices.face_selector_age_range[0]
			face_selector_age_end = state_manager.get_item('face_selector_age_end') or facefusion.choices.face_selector_age_range[-1]
			FACE_SCANNER_AGE_RANGE_SLIDER = RangeSlider(
				label = translator.get('uis.face_selector_age_range_slider'),
				minimum = facefusion.choices.face_selector_age_range[0],
				maximum = facefusion.choices.face_selector_age_range[-1],
				value = (face_selector_age_start, face_selector_age_end),
				step = calculate_int_step(facefusion.choices.face_selector_age_range)
			)
	FACE_SCANNER_REFERENCE_FACE_DISTANCE_SLIDER = gradio.Slider(
		label = 'REFERENCE FACE DISTANCE',
		value = state_manager.get_item('reference_face_distance'),
		step = calculate_float_step(facefusion.choices.reference_face_distance_range),
		minimum = facefusion.choices.reference_face_distance_range[0],
		maximum = facefusion.choices.reference_face_distance_range[-1]
	)


def listen() -> None:
	FACE_SCANNER_SCAN_BUTTON.click(scan, inputs = [ FACE_SCANNER_SCAN_EVERY_SLIDER, FACE_SCANNER_MIN_FACE_SIZE_SLIDER, FACE_SCANNER_MAX_FACES_SLIDER ], outputs = [ FACE_SCANNER_GALLERY, FACE_SCANNER_STATUS, FACE_SCANNER_TEST_IMAGE ])
	FACE_SCANNER_GALLERY.select(select_face, outputs = FACE_SCANNER_STATUS)
	FACE_SCANNER_TEST_BUTTON.click(test_face, outputs = [ FACE_SCANNER_TEST_IMAGE, FACE_SCANNER_STATUS ])
	FACE_SCANNER_REFERENCE_FACE_DISTANCE_SLIDER.release(update_reference_face_distance, inputs = FACE_SCANNER_REFERENCE_FACE_DISTANCE_SLIDER, outputs = FACE_SCANNER_STATUS)

	scan_outputs = [ FACE_SCANNER_GALLERY, FACE_SCANNER_STATUS, FACE_SCANNER_TEST_IMAGE ]
	FACE_SCANNER_ORDER_DROPDOWN.change(update_face_selector_order, inputs = FACE_SCANNER_ORDER_DROPDOWN, outputs = scan_outputs)
	FACE_SCANNER_GENDER_DROPDOWN.change(update_face_selector_gender, inputs = FACE_SCANNER_GENDER_DROPDOWN, outputs = scan_outputs)
	FACE_SCANNER_RACE_DROPDOWN.change(update_face_selector_race, inputs = FACE_SCANNER_RACE_DROPDOWN, outputs = scan_outputs)
	FACE_SCANNER_AGE_RANGE_SLIDER.release(update_face_selector_age_range, inputs = FACE_SCANNER_AGE_RANGE_SLIDER, outputs = scan_outputs)

	for ui_component in get_ui_components(
	[
		'target_image',
		'target_video'
	]):
		for method in [ 'change', 'clear' ]:
			getattr(ui_component, method)(clear_scan, outputs = scan_outputs)


def clear_scan() -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	return invalidate_scan('Target changed')


def invalidate_scan(reason : str) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	"""A filter or a new target moves faces around, so a position picked from the old scan would point at somebody else."""
	global FACE_CANDIDATES
	global SELECTED_CANDIDATE_INDEX

	had_candidates = bool(FACE_CANDIDATES)
	FACE_CANDIDATES = []
	SELECTED_CANDIDATE_INDEX = None
	state_manager.set_item('reference_frame_number', 0)
	state_manager.set_item('reference_face_position', 0)
	status = '{} — press **SCAN FOR FACES** again.'.format(reason) if had_candidates else IDLE_STATUS
	return gradio.Gallery(value = None), gradio.Markdown(value = status), gradio.Image(value = None, visible = False)


def update_face_selector_order(face_selector_order : FaceSelectorOrder) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	state_manager.set_item('face_selector_order', convert_str_none(face_selector_order))
	return invalidate_scan('Face order changed')


def update_face_selector_gender(face_selector_gender : Gender) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	state_manager.set_item('face_selector_gender', convert_str_none(face_selector_gender))
	return invalidate_scan('Gender filter changed')


def update_face_selector_race(face_selector_race : Race) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	state_manager.set_item('face_selector_race', convert_str_none(face_selector_race))
	return invalidate_scan('Race filter changed')


def update_face_selector_age_range(face_selector_age_range : Tuple[float, float]) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	face_selector_age_start, face_selector_age_end = face_selector_age_range
	state_manager.set_item('face_selector_age_start', int(face_selector_age_start))
	state_manager.set_item('face_selector_age_end', int(face_selector_age_end))
	return invalidate_scan('Age filter changed')


def scan(scan_every : float, min_face_size : float, max_faces : float, progress : gradio.Progress = gradio.Progress()) -> Tuple[gradio.Gallery, gradio.Markdown, gradio.Image]:
	global FACE_CANDIDATES
	global SELECTED_CANDIDATE_INDEX
	global VIDEO_FPS

	target_path = state_manager.get_item('target_path')

	if not is_image(target_path) and not is_video(target_path):
		return gradio.Gallery(value = None), gradio.Markdown(value = 'Choose a target image or video first.'), gradio.Image(value = None, visible = False)

	# the scan must see exactly what the swap will see
	state_manager.set_item('face_selector_mode', 'reference')
	clear_static_faces()
	SELECTED_CANDIDATE_INDEX = None
	VIDEO_FPS = detect_video_fps(target_path) or 25.0

	if is_image(target_path):
		FACE_CANDIDATES = scan_target_image(target_path)
		notes : List[str] = []
	else:
		frame_numbers = create_frame_numbers(target_path, VIDEO_FPS, scan_every, 0.0, None, MAX_SAMPLES)

		def report_progress(sample_index : int, sample_total : int, candidate_total : int) -> None:
			progress(sample_index / sample_total, desc = 'scanning frame {} of {}, {} distinct face(s) so far'.format(sample_index, sample_total, candidate_total))

		FACE_CANDIDATES = scan_video(target_path, frame_numbers, CLUSTER_DISTANCE, report_progress)
		FACE_CANDIDATES, notes = filter_candidates(FACE_CANDIDATES, 1, min_face_size / 100, int(max_faces))

	if not FACE_CANDIDATES:
		return gradio.Gallery(value = None), gradio.Markdown(value = 'No faces found. Lower FACE DETECTOR SCORE, or scan more often.'), gradio.Image(value = None, visible = False)

	status = '**{} distinct face(s) found.** Click the one you want to swap.'.format(len(FACE_CANDIDATES))

	for note in notes:
		status += '  \n_{}_'.format(note)

	return gradio.Gallery(value = create_gallery_items(FACE_CANDIDATES)), gradio.Markdown(value = status), gradio.Image(value = None, visible = False)


def scan_target_image(target_path : str) -> List[FaceCandidate]:
	"""An image is a single frame, so every detected face is its own candidate."""
	candidates : List[FaceCandidate] = []
	target_vision_frame = read_static_image(target_path)
	faces = sort_and_filter_faces(get_many_faces([ target_vision_frame ]))

	for face_position, face in enumerate(faces):
		candidates.append(FaceCandidate(
			embedding_sum = face.embedding_norm,
			appearance_count = 1,
			reference_face_position = face_position,
			reference_crop = create_face_crop(target_vision_frame, face),
			gender = str(face.gender),
			age = format_age(face.age),
			race = str(face.race)
		))

	return candidates


def create_gallery_items(candidates : List[FaceCandidate]) -> List[Tuple[numpy.ndarray[Any, Any], str]]:
	gallery_items = []

	for candidate_index, candidate in enumerate(candidates):
		crop_vision_frame = cv2.cvtColor(candidate.reference_crop, cv2.COLOR_BGR2RGB)
		gallery_items.append((crop_vision_frame, create_gallery_caption(candidate_index, candidate)))

	return gallery_items


def create_gallery_caption(candidate_index : int, candidate : FaceCandidate) -> str:
	"""Kept short on purpose, a tile is only a few characters wide before it ellipsises."""
	if candidate.appearance_count > 1:
		return '#{} · {} hits'.format(candidate_index, candidate.appearance_count)
	return '#{} · {}'.format(candidate_index, candidate.gender)


def select_face(event : gradio.SelectData) -> gradio.Markdown:
	global SELECTED_CANDIDATE_INDEX

	if event.index is None or event.index >= len(FACE_CANDIDATES):
		return gradio.Markdown(value = IDLE_STATUS)

	SELECTED_CANDIDATE_INDEX = event.index
	candidate = FACE_CANDIDATES[event.index]
	state_manager.set_item('face_selector_mode', 'reference')
	state_manager.set_item('reference_frame_number', candidate.reference_frame_number)
	state_manager.set_item('reference_face_position', candidate.reference_face_position)
	return gradio.Markdown(value = create_selection_status())


def create_selection_status() -> str:
	if SELECTED_CANDIDATE_INDEX is None:
		return IDLE_STATUS

	candidate = FACE_CANDIDATES[SELECTED_CANDIDATE_INDEX]
	status = '**Face #{} locked in** — {} {}, seen {}× between {} and {}. Reference frame {}, position {}. Press START to swap it through the whole video.'.format(
		SELECTED_CANDIDATE_INDEX,
		candidate.gender,
		candidate.age,
		candidate.appearance_count,
		format_timestamp(candidate.first_frame_number, VIDEO_FPS),
		format_timestamp(candidate.last_frame_number, VIDEO_FPS),
		candidate.reference_frame_number,
		candidate.reference_face_position)

	for other_index, identity_distance in find_similar_candidates(FACE_CANDIDATES, SELECTED_CANDIDATE_INDEX, state_manager.get_item('reference_face_distance')):
		status += '  \n⚠ face #{} is only {:.2f} away and would be swapped too — drop REFERENCE FACE DISTANCE below that.'.format(other_index, identity_distance)

	return status


def update_reference_face_distance(reference_face_distance : float) -> gradio.Markdown:
	state_manager.set_item('reference_face_distance', reference_face_distance)
	return gradio.Markdown(value = create_selection_status())


def test_face(progress : gradio.Progress = gradio.Progress()) -> Tuple[gradio.Image, gradio.Markdown]:
	from facefusion.uis.components.preview import process_preview_frame

	if SELECTED_CANDIDATE_INDEX is None:
		return gradio.Image(value = None, visible = False), gradio.Markdown(value = 'Pick a face from the gallery first.')

	target_path = state_manager.get_item('target_path')
	progress(0.0, desc = 'rendering one frame')
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	reference_frame_number = state_manager.get_item('reference_frame_number')

	if source_audio_path and state_manager.get_item('output_video_fps') and reference_frame_number:
		temp_voice_frame = get_voice_frame(source_audio_path, state_manager.get_item('output_video_fps'), reference_frame_number)
		if numpy.any(temp_voice_frame):
			source_voice_frame = temp_voice_frame

	if is_image(target_path):
		target_vision_frame = read_static_image(target_path)
	else:
		target_vision_frame = read_video_frame(target_path, reference_frame_number)

	if target_vision_frame is None:
		return gradio.Image(value = None, visible = False), gradio.Markdown(value = 'Could not read the reference frame.')

	comparison_vision_frame = process_preview_frame(target_vision_frame, source_vision_frames, source_audio_frame, source_voice_frame, target_vision_frame, 'frame-by-frame', uis_choices.preview_resolutions[-1])
	progress(1.0, desc = 'done')
	return gradio.Image(value = cv2.cvtColor(comparison_vision_frame, cv2.COLOR_BGR2RGB), visible = True), gradio.Markdown(value = create_selection_status())
