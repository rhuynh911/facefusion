# tools

Helper scripts that sit on top of FaceFusion. Run them from the repo root inside the same
conda environment you use for `facefusion.py`.

Both entry points below do the same thing — scan the target, group the faces, let you pick
one, swap it through the whole video — one in the browser, one on the command line. They
share the scanning code in `facefusion/face_scanner.py`, so they always agree on what they
find.

## Web UI: the `auto_swap` layout

```bash
python facefusion.py run --ui-layouts auto_swap
```

Same FaceFusion UI you already use, with the PREVIEW panel, the PREVIEW FRAME slider and the
old reference-face gallery replaced by a face scanner. Everything else — processors, models,
masks, execution providers, output settings — is exactly where it was.

The flow:

1. Drop your portraits in **SOURCE** and the video in **TARGET**, as usual.
2. Press **SCAN FOR FACES**. A progress bar walks the video; every distinct person shows up
   in the **FACES IN THE TARGET** gallery, captioned `#0 · 21 hits · 00:05-02:15`, most
   frequent first.
3. Click the face you want. The status line confirms `Face #3 locked in — reference frame
   3361, position 0`, and warns you if another person is similar enough to be swapped too.
4. Optionally press **TEST THIS FACE** to render one before/after frame.
5. Press **START**. FaceFusion swaps that person through the whole video.

Scan controls sit under the gallery:

| Control | Why you'd touch it |
| --- | --- |
| `SCAN EVERY (SECONDS)` | Lower it when someone only appears in a short shot. Higher is faster. |
| `MINIMUM FACE SIZE (% OF FRAME)` | Set to 0 to reveal background extras that are hidden by default. |
| `MAXIMUM FACES SHOWN` | How many faces the gallery holds. |
| `REFERENCE FACE DISTANCE` | Tighten it when two people in the video look alike. |

Changing the target clears the scan, so you never pick a face from a stale gallery. The stock
UI is untouched — `python facefusion.py run` still opens the original layout.

## CLI: auto_swap.py

Swaps one person in a video without touching the web UI. It replaces the manual loop of
loading the target, dragging the PREVIEW FRAME slider until the right face appears, and
clicking it in the REFERENCE FACE gallery.

```bash
python tools/auto_swap.py -s portraits/ -t video.mp4
```

What it does:

1. **Scans** the video at `--scan-every` seconds, detecting faces with the same detector,
   landmarker and recognizer settings the swap itself will use.
2. **Groups** the detections into distinct people by comparing face embeddings, using the
   same distance metric as `face_selector.compare_faces`.
3. **Shows** them as a numbered contact sheet (`<target>_faces/faces_preview.png`, opened
   automatically) plus a table, and asks which one to swap.
4. **Runs** `facefusion.py headless-run` with the `--reference-frame-number` and
   `--reference-face-position` that resolve to the person you picked, choosing the largest,
   most frontal, least ambiguous frame it saw as the reference.

### Options that matter

| Option | Why you'd touch it |
| --- | --- |
| `--scan-every 0.5` | Denser scan. Use it when someone only appears in a short shot. |
| `--min-face-height 0` | Keep background extras that are filtered out by default. |
| `--reference-face-distance 0.2` | Tighten matching when two people in the video look alike. The script warns when this is a risk. |
| `--face-index 2` | Skip the prompt, e.g. when re-running after a settings change. |
| `--scan-only` | Just report the faces, swap nothing. `-s` is not required for this. |
| `--dry-run` | Print the `headless-run` command instead of executing it. |

Anything after a bare `--` is forwarded verbatim to `headless-run` *and* to the scan, so both
stay in sync:

```bash
python tools/auto_swap.py -s portraits/ -t video.mp4 -- \
	--face-swapper-model hyperswap_1a_256 --face-enhancer-model gfpgan_1.4 --execution-providers cuda
```

### Using other processors

`--processors` takes the same names as the web UI, and each processor's own settings go after
the `--`. The face you pick in the contact sheet is the face every one of them acts on.

**face_swapper** (default) — the knobs from the PROCESSORS panel:

```bash
python tools/auto_swap.py -s portraits/ -t video.mp4 -- \
	--face-swapper-model hyperswap_1a_256 --face-swapper-pixel-boost 256x256 --face-swapper-weight 0.5
```

**face_enhancer** — restores the swapped face; run it after the swapper:

```bash
python tools/auto_swap.py -s portraits/ -t video.mp4 --processors face_swapper face_enhancer -- \
	--face-enhancer-model gfpgan_1.4 --face-enhancer-blend 80 --face-enhancer-weight 0.5
```

**lip_syncer** — drives the mouth from an audio track, so it **needs an audio file in
`--source-paths`** alongside the portraits. The script passes audio sources through and
refuses to start without one:

```bash
python tools/auto_swap.py -s portraits/ voice.mp3 -t video.mp4 \
	--processors face_swapper lip_syncer -- \
	--lip-syncer-model wav2lip_gan_96 --lip-syncer-weight 0.5
```

**deep_swapper** — uses a pretrained per-identity model instead of your portraits, so
`--deep-swapper-model` decides whose face appears; the `-s` portraits are ignored by this
processor (they still feed `face_swapper` if you run both). It also requires the output
extension to match the target's:

```bash
python tools/auto_swap.py -s portraits/ -t video.mp4 -o out.mp4 \
	--processors deep_swapper -- \
	--deep-swapper-model iperov/elon_musk_224 --deep-swapper-morph 100
```

To mirror the full GUI setup from all three at once:

```bash
python tools/auto_swap.py -s portraits/ voice.mp3 -t video.mp4 -o out.mp4 \
	--processors face_swapper lip_syncer deep_swapper -- \
	--face-swapper-model hyperswap_1a_256 --face-swapper-pixel-boost 256x256 \
	--lip-syncer-model wav2lip_gan_96 --deep-swapper-model iperov/elon_musk_224
```

Processors run in the order given. Run `python facefusion.py headless-run --help` for the
full option list of any processor.

### Notes

- FaceFusion swaps one identity per run. To replace a second person, run again with the first
  run's output as the target. This is a FaceFusion limit, not a limit of these tools.
- The contact sheet, per-face crops and a `faces.json` (frame numbers, positions, timestamps)
  are written to `<target>_faces/`, so a scan can be reused for scripting.
- Face ordering must match between the scan and the swap, so the script passes
  `--face-selector-order` explicitly. It warns if a gender/race/age filter in `facefusion.ini`
  is hiding faces from both.

## How the pick is turned into a swap

Both entry points do the same trick. FaceFusion's `reference` selector mode identifies a face
by two numbers: `--reference-frame-number` (which frame to look at) and
`--reference-face-position` (which face in that frame, after ordering and filtering). Dragging
the PREVIEW FRAME slider and clicking a thumbnail is just a manual way of producing those two
numbers.

The scanner produces them for you: it samples the video, clusters the detections into people,
and for each person keeps the frame where that face is largest, most frontal and least likely
to be confused with a same-sized neighbour. Picking a face writes that pair into state
(`facefusion/face_scanner.py` does the work, `facefusion/uis/components/face_scanner.py` and
`tools/auto_swap.py` are the two front ends), and the normal FaceFusion pipeline takes it from
there unchanged.
