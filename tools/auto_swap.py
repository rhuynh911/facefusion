#!/usr/bin/env python3
"""
Interactive face swap driver for FaceFusion.

Replaces the manual "drag the PREVIEW FRAME slider until the right face shows up" dance:

	1. scans the target video and groups every detected face into distinct people
	2. shows them as a numbered contact sheet and asks which one to swap
	3. runs `facefusion.py headless-run` with the matching --reference-frame-number
	   and --reference-face-position for the person you picked

The same scan powers the web UI, see `python facefusion.py run --ui-layouts auto_swap`.

Usage:
	python tools/auto_swap.py -s portraits/ -t video.mp4
	python tools/auto_swap.py -s a.jpg b.jpg -t video.mp4 -o out.mp4 --face-index 1
	python tools/auto_swap.py -s portraits/ -t video.mp4 -- --face-swapper-model hyperswap_1a_256
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import cv2  # noqa: E402
import numpy  # noqa: E402

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from facefusion import conda, face_classifier, face_detector, face_landmarker, face_recognizer, logger, state_manager  # noqa: E402
from facefusion.args import apply_args  # noqa: E402
from facefusion.common_helper import is_windows  # noqa: E402
from facefusion.face_scanner import CROP_SIZE, FaceCandidate, create_frame_numbers, filter_candidates, find_similar_candidates, format_timestamp, scan_video  # noqa: E402
from facefusion.filesystem import create_directory, get_file_name, has_audio, has_image, is_audio, is_directory, is_image, is_video, resolve_file_paths, resolve_file_pattern  # noqa: E402
from facefusion.program import create_program  # noqa: E402
from facefusion.vision import count_video_frame_total, detect_video_fps, write_image  # noqa: E402

TILE_COLUMNS = 5
LABEL_HEIGHT = 34


def create_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog = 'auto_swap',
		description = 'Scan a video, pick one face interactively, swap it with your portraits.',
		epilog = 'Anything after a bare -- is forwarded verbatim to facefusion.py headless-run (and to the scan, so both stay in sync).',
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-s', '--source-paths', help = 'portrait image(s), a directory of them, or a glob (not needed with --scan-only)', nargs = '+', default = [])
	parser.add_argument('-t', '--target-path', help = 'video that contains the face to replace', required = True)
	parser.add_argument('-o', '--output-path', help = 'output video (default: <target>-swapped.<ext> next to the target)')
	parser.add_argument('--processors', help = 'processors to run', nargs = '+', default = [ 'face_swapper' ])
	parser.add_argument('--face-index', help = 'skip the prompt and take this face from the scan', type = int)
	parser.add_argument('--scan-every', help = 'seconds between scanned frames', type = float, default = 1.0)
	parser.add_argument('--scan-start', help = 'start scanning at this second', type = float, default = 0.0)
	parser.add_argument('--scan-end', help = 'stop scanning at this second', type = float)
	parser.add_argument('--max-samples', help = 'hard cap on scanned frames', type = int, default = 400)
	parser.add_argument('--min-appearances', help = 'drop people seen in fewer scanned frames than this', type = int, default = 1)
	parser.add_argument('--min-face-height', help = 'drop faces never taller than this fraction of the frame, i.e. background extras (0 keeps everything)', type = float, default = 0.05)
	parser.add_argument('--max-faces', help = 'most faces to put on the contact sheet', type = int, default = 24)
	parser.add_argument('--cluster-distance', help = 'identity grouping threshold, same scale as --reference-face-distance', type = float, default = 0.3)
	parser.add_argument('--reference-face-distance', help = 'how loosely facefusion matches the chosen face across the video', type = float, default = 0.3)
	parser.add_argument('--face-selector-order', help = 'face ordering used for both the scan and the run', default = 'large-small')
	parser.add_argument('--preview-dir', help = 'where to write the contact sheet and crops (default: <target>_faces next to the target)')
	parser.add_argument('--no-open', help = 'do not open the contact sheet in an image viewer', action = 'store_true')
	parser.add_argument('--scan-only', help = 'scan and report the faces, then stop', action = 'store_true')
	parser.add_argument('--dry-run', help = 'print the headless-run command instead of running it', action = 'store_true')
	return parser


def split_extra_args(argv : List[str]) -> Tuple[List[str], List[str]]:
	if '--' in argv:
		separator_index = argv.index('--')
		return argv[:separator_index], argv[separator_index + 1:]
	return argv, []


def resolve_source_paths(source_paths : List[str]) -> List[str]:
	resolved_paths : List[str] = []

	for source_path in source_paths:
		if is_directory(source_path):
			resolved_paths.extend(resolve_file_paths(source_path))
		elif is_image(source_path):
			resolved_paths.append(source_path)
		else:
			resolved_paths.extend(resolve_file_pattern(source_path))

	# audio is kept as well, lip_syncer refuses to run without a voice source
	return [ resolved_path for resolved_path in resolved_paths if is_image(resolved_path) or is_audio(resolved_path) ]


def validate_source_paths(source_paths : List[str], requested_paths : List[str], processors : List[str]) -> bool:
	if not source_paths:
		print('  ! no usable sources found in: {}'.format(' '.join(requested_paths) or '(none given, pass -s)'))
		return False

	if 'face_swapper' in processors and not has_image(source_paths):
		print('  ! face_swapper needs at least one portrait image in --source-paths')
		return False

	if 'lip_syncer' in processors and not has_audio(source_paths):
		print('  ! lip_syncer needs an audio file in --source-paths, e.g. -s portraits/ voice.mp3')
		return False

	return True


def bootstrap_state(target_path : str, face_selector_order : str, reference_face_distance : float, extra_args : List[str]) -> None:
	program = create_program()
	program_args =\
	[
		'headless-run',
		'--target-path', target_path,
		'--face-selector-mode', 'reference',
		'--face-selector-order', face_selector_order,
		'--reference-face-distance', str(reference_face_distance)
	]
	args = vars(program.parse_args(program_args + extra_args))
	apply_args(args, state_manager.init_item)
	logger.init(state_manager.get_item('log_level'))


def pre_check() -> bool:
	return all(module.pre_check() for module in [ face_classifier, face_detector, face_landmarker, face_recognizer ])


def render_progress(sample_index : int, sample_total : int, candidate_total : int) -> None:
	percent = int(sample_index / sample_total * 100)
	sys.stdout.write('\r  scanning {}/{} frames ({}%) - {} distinct face(s) so far '.format(sample_index, sample_total, percent, candidate_total))
	sys.stdout.flush()


def create_contact_sheet(candidates : List[FaceCandidate], video_fps : float) -> numpy.ndarray[Any, Any]:
	column_total = min(TILE_COLUMNS, len(candidates))
	row_total = math.ceil(len(candidates) / column_total)
	sheet_height = row_total * (CROP_SIZE + LABEL_HEIGHT)
	sheet_width = column_total * CROP_SIZE
	contact_sheet = numpy.full((sheet_height, sheet_width, 3), 24, dtype = numpy.uint8)

	for candidate_index, candidate in enumerate(candidates):
		row_index = candidate_index // column_total
		column_index = candidate_index % column_total
		start_x = column_index * CROP_SIZE
		start_y = row_index * (CROP_SIZE + LABEL_HEIGHT)

		if candidate.reference_crop is not None:
			contact_sheet[start_y:start_y + CROP_SIZE, start_x:start_x + CROP_SIZE] = candidate.reference_crop

		cv2.rectangle(contact_sheet, (start_x, start_y), (start_x + CROP_SIZE - 1, start_y + CROP_SIZE - 1), (64, 64, 64), 1)
		label = '[{}]  {} hits  {}-{}'.format(candidate_index, candidate.appearance_count, format_timestamp(candidate.first_frame_number, video_fps), format_timestamp(candidate.last_frame_number, video_fps))
		cv2.putText(contact_sheet, label, (start_x + 8, start_y + CROP_SIZE + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

	return contact_sheet


def render_candidate_table(candidates : List[FaceCandidate], video_fps : float) -> None:
	print('\n  #   hits   first    last     size   frame   pos   gender   age     race')
	print('  ' + '-' * 73)

	for candidate_index, candidate in enumerate(candidates):
		print('  {:<3} {:<6} {:<8} {:<8} {:<6} {:<7} {:<5} {:<8} {:<7} {}'.format(
			candidate_index,
			candidate.appearance_count,
			format_timestamp(candidate.first_frame_number, video_fps),
			format_timestamp(candidate.last_frame_number, video_fps),
			'{:.0f}%'.format(candidate.max_face_height * 100),
			candidate.reference_frame_number,
			candidate.reference_face_position,
			candidate.gender,
			candidate.age,
			candidate.race))


def warn_similar_candidates(candidates : List[FaceCandidate], candidate_index : int, reference_face_distance : float) -> None:
	for other_index, identity_distance in find_similar_candidates(candidates, candidate_index, reference_face_distance):
		print('  ! face [{}] sits only {:.2f} away from face [{}] - facefusion may swap both.'.format(candidate_index, identity_distance, other_index))
		print('    lower it with --reference-face-distance {:.2f}'.format(max(0.05, identity_distance - 0.05)))


def write_candidate_assets(candidates : List[FaceCandidate], preview_dir : str, video_fps : float) -> str:
	create_directory(preview_dir)
	contact_sheet_path = os.path.join(preview_dir, 'faces_preview.png')
	write_image(contact_sheet_path, create_contact_sheet(candidates, video_fps))
	candidate_reports : List[Dict[str, Any]] = []

	for candidate_index, candidate in enumerate(candidates):
		if candidate.reference_crop is not None:
			write_image(os.path.join(preview_dir, 'face_{:02d}.jpg'.format(candidate_index)), candidate.reference_crop)

		candidate_reports.append(
		{
			'face_index': candidate_index,
			'appearance_count': candidate.appearance_count,
			'first_frame_number': candidate.first_frame_number,
			'last_frame_number': candidate.last_frame_number,
			'first_timestamp': format_timestamp(candidate.first_frame_number, video_fps),
			'last_timestamp': format_timestamp(candidate.last_frame_number, video_fps),
			'reference_frame_number': candidate.reference_frame_number,
			'reference_face_position': candidate.reference_face_position,
			'max_face_height': round(candidate.max_face_height, 4),
			'sample_frame_numbers': sorted(candidate.frame_numbers),
			'gender': candidate.gender,
			'age': candidate.age,
			'race': candidate.race
		})

	with open(os.path.join(preview_dir, 'faces.json'), 'w', encoding = 'utf-8') as report_file:
		json.dump(candidate_reports, report_file, indent = 2)

	return contact_sheet_path


def open_file(file_path : str) -> None:
	try:
		if is_windows():
			os.startfile(file_path)  # type:ignore[attr-defined]
		elif sys.platform == 'darwin':
			subprocess.run([ 'open', file_path ], check = False)
		else:
			subprocess.run([ 'xdg-open', file_path ], check = False)
	except OSError:
		pass


def prompt_face_index(candidate_total : int) -> Optional[int]:
	while True:
		answer = input('\n  Which face do you want to swap? [0-{}, Enter = 0, q = quit]: '.format(candidate_total - 1)).strip().lower()

		if answer in [ 'q', 'quit', 'exit' ]:
			return None
		if not answer:
			return 0
		if answer.isdigit() and int(answer) < candidate_total:
			return int(answer)

		print('  not a valid choice')


def suggest_output_path(target_path : str) -> str:
	directory_path = os.path.dirname(os.path.abspath(target_path))
	file_name = get_file_name(target_path)
	_, file_extension = os.path.splitext(target_path)
	return os.path.join(directory_path, file_name + '-swapped' + file_extension)


def build_run_command(args : argparse.Namespace, source_paths : List[str], output_path : str, candidate : FaceCandidate, extra_args : List[str]) -> List[str]:
	run_command =\
	[
		sys.executable, 'facefusion.py', 'headless-run',
		'--source-paths', *source_paths,
		'--target-path', args.target_path,
		'--output-path', output_path,
		'--processors', *args.processors,
		'--face-selector-mode', 'reference',
		'--face-selector-order', args.face_selector_order,
		'--reference-face-position', str(candidate.reference_face_position),
		'--reference-face-distance', str(args.reference_face_distance),
		'--reference-frame-number', str(candidate.reference_frame_number)
	]
	run_command.extend(extra_args)
	return run_command


def warn_conflicting_extra_args(extra_args : List[str]) -> None:
	conflicting_args = [ '--reference-frame-number', '--reference-face-position', '--face-selector-mode' ]

	for extra_arg in extra_args:
		if extra_arg in conflicting_args:
			print('  ! {} in the passthrough args overrides what the scan picked'.format(extra_arg))


def warn_active_filters() -> None:
	if state_manager.get_item('face_selector_gender') or state_manager.get_item('face_selector_race') or state_manager.get_item('face_selector_age_start') or state_manager.get_item('face_selector_age_end'):
		print('  ! a gender/race/age filter is active in facefusion.ini - filtered-out faces are invisible to both the scan and the swap')


def main() -> int:
	argv, extra_args = split_extra_args(sys.argv[1:])
	args = create_parser().parse_args(argv)
	args.target_path = os.path.abspath(args.target_path)

	if not is_video(args.target_path):
		print('  ! target is not a readable video: {}'.format(args.target_path))
		return 1

	source_paths = [ os.path.abspath(source_path) for source_path in resolve_source_paths(args.source_paths) ]

	if not args.scan_only and not validate_source_paths(source_paths, args.source_paths, args.processors):
		return 1

	if args.output_path:
		args.output_path = os.path.abspath(args.output_path)
	if args.preview_dir:
		args.preview_dir = os.path.abspath(args.preview_dir)

	# facefusion resolves facefusion.ini, .assets and its processor modules relative to the repo root
	os.chdir(ROOT_PATH)
	conda.setup()
	bootstrap_state(args.target_path, args.face_selector_order, args.reference_face_distance, extra_args)

	if not pre_check():
		print('  ! face analyser models are unavailable')
		return 1

	video_fps = detect_video_fps(args.target_path) or 25.0
	frame_numbers = create_frame_numbers(args.target_path, video_fps, args.scan_every, args.scan_start, args.scan_end, args.max_samples)

	if not frame_numbers:
		print('  ! nothing to scan in the requested range')
		return 1

	source_image_total = len([ source_path for source_path in source_paths if is_image(source_path) ])
	source_audio_total = len(source_paths) - source_image_total
	print('\n  sources  : {} portrait(s), {} audio'.format(source_image_total, source_audio_total) if source_paths else '\n  sources  : none (scan only)')
	print('  target   : {} ({} frames @ {:.2f} fps)'.format(args.target_path, count_video_frame_total(args.target_path), video_fps))
	print('  sampling : every {} frame(s), {} samples\n'.format(max(1, frame_numbers[1] - frame_numbers[0]) if len(frame_numbers) > 1 else 1, len(frame_numbers)))
	warn_active_filters()

	candidates = scan_video(args.target_path, frame_numbers, args.cluster_distance, render_progress)
	sys.stdout.write('\n')

	if not candidates:
		print('  ! no faces detected - try a smaller --scan-every or a lower --face-detector-score')
		return 1

	candidates, notes = filter_candidates(candidates, args.min_appearances, args.min_face_height, args.max_faces)

	for note in notes:
		print('  {}'.format(note))

	preview_dir = args.preview_dir or os.path.join(os.path.dirname(os.path.abspath(args.target_path)), get_file_name(args.target_path) + '_faces')
	contact_sheet_path = write_candidate_assets(candidates, preview_dir, video_fps)
	render_candidate_table(candidates, video_fps)
	print('\n  preview  : {}'.format(contact_sheet_path))

	if not args.no_open:
		open_file(contact_sheet_path)

	if args.scan_only:
		return 0

	if args.face_index is not None:
		if args.face_index >= len(candidates):
			print('  ! --face-index {} is out of range'.format(args.face_index))
			return 1
		candidate_index : Optional[int] = args.face_index
	elif sys.stdin.isatty():
		candidate_index = prompt_face_index(len(candidates))
	else:
		print('  ! not a terminal - rerun with --face-index N')
		return 1

	if candidate_index is None:
		print('  aborted')
		return 0

	candidate = candidates[candidate_index]
	warn_similar_candidates(candidates, candidate_index, args.reference_face_distance)
	warn_conflicting_extra_args(extra_args)
	output_path = args.output_path or suggest_output_path(args.target_path)
	run_command = build_run_command(args, source_paths, output_path, candidate, extra_args)

	print('\n  face [{}] -> reference frame {}, position {}'.format(candidate_index, candidate.reference_frame_number, candidate.reference_face_position))
	print('  output   : {}\n'.format(output_path))
	print('  ' + subprocess.list2cmdline(run_command) + '\n')

	if args.dry_run:
		return 0

	sys.stdout.flush()
	return subprocess.run(run_command, cwd = ROOT_PATH, check = False).returncode


if __name__ == '__main__':
	sys.exit(main())
