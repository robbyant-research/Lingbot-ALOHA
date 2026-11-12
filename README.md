> Local opening is supported: double-click **index.html**. The original 3D animation and models are bundled in **halo-viewer.js**. Rebuild after editing the source viewer using `python tools/build_local_viewer.py`.

> Final local version: see **FINAL_NOTES.md** and **assets-manifest.json** for the current assets and corrected cut timings. The preview notes below are retained as reference.

# HALO project page

Static project page for **HALO: Human-Aligned Data Collection for Whole-Body Mobile Manipulation**, built on the XU Lab "page" template (plain HTML + CSS + vanilla JS, no build step). The "Mechanical Design" section is an interactive three.js viewer driven by scrolling.

```
index.html              the page
style.css, page.js      template files (+ HALO additions at the end of style.css)
halo3d.js               scroll-driven 3D viewer (framework, wheels, steps, labels, camera)
halo-models.js          loader for the mesh bundles in assets/models/
vendor/three/           three.js r169 (MIT) + OrbitControls + RoomEnvironment, no CDN needed
assets/
  video/                13 clips cut from the supplementary video + the full video
  posters/              one poster frame per video (same file name, .jpg)
  figures/              figures cropped from the paper PDF
  models/               x5_arm.{json,bin}, tracer2.{json,bin}  (3D mesh bundles)
  paper/HALO.pdf        the paper
tools/
  cut_clips.sh          re-cut all clips/posters from a better master video
  build_arm_bundle.py   ARX5 arm + HALO gripper + cameras  -> assets/models/x5_arm.*
  build_tracer_bundle.py  AgileX Tracer 2.0                 -> assets/models/tracer2.*
```

## Preview locally

The 3D viewer loads its models with `fetch`, which browsers block on `file://`. Serve the folder:

```sh
cd halo-project-page
python3 -m http.server 8000      # open http://localhost:8000/
```

On GitHub Pages (or any static host) it works as-is; keep the folder structure and relative paths. Without WebGL, or if the model files cannot be loaded, the stage shows the Fig. 4 photo instead.

## Fill in before publishing

Search `index.html` for `TODO`. Placeholders are drawn with a dashed yellow outline so they are hard to miss.

| What | Where in `index.html` |
|---|---|
| Author names (with links/superscripts) | `hero__authors` |
| Affiliations | `hero__affil` |
| arXiv / Code links | commented-out pills under the hero (never leave an empty `href`) |
| BibTeX | `#bib-text` (author list, venue, year) |
| Paper link | the Paper pill points to `assets/paper/HALO.pdf` (the anonymized PDF). Replace it with the arXiv link when public, or swap the file. |
| `<title>` / `<meta name="description">` | top of the file |

The footer (`XU Lab · HKUST`) is required by the template; change it only if this page is not hosted by the lab.

## Replacing the videos

All clips were cut from **`final_v2_x264_preview.mp4`**, a 720 × 480 anamorphic preview (displayed 16:9). They are re-encoded to square-pixel 854 × 480 H.264, muted, `+faststart`. Replace them with full-resolution versions before publishing.

### Option A: re-cut everything automatically (recommended)

If your full-resolution master has the same edit as the preview:

```sh
bash tools/cut_clips.sh /path/to/supplementary_master.mp4 720    # or 1080
```

This overwrites `assets/video/*.mp4` and `assets/posters/*.jpg` with the same file names, so `index.html` needs no change. If your master is edited differently, adjust the start/end times in the table inside the script.

### Option B: replace files by hand

Keep the file name, or update the `data-src` / `poster` attributes in `index.html`. Each video has a matching poster image (`assets/posters/<name>.jpg`, any frame of the clip).

| File (`assets/video/`) | Source time in the preview | Content | Used in |
|---|---|---|---|
| `teaser.mp4` | 0:00.0 – 0:13.1 | intro montage | hero (autoplays) |
| `collect_operator.mp4` | 0:13.4 – 0:21.9 | operator collecting at the shelf | Collecting Demonstrations |
| `collect_headcam.mp4` | 0:25.0 – 0:30.4 | head-camera view, inset after hand removal | Collecting Demonstrations |
| `design_cad.mp4` | 0:31.7 – 0:39.2 | render: collection system → deployment system | Mechanical Design (below the viewer) |
| `lift_box.mp4` | 0:40.5 – 0:47.6 | Lift Box, 1.3× | Policy Rollouts |
| `push_chair.mp4` | 0:47.9 – 1:00.0 | Push Chair, 1× | Policy Rollouts (first) |
| `push_chair_spatial.mp4` | 1:00.3 – 1:12.6 | Push Chair, spatial generalization | What the Policies Learn |
| `push_chair_env.mp4` | 1:12.9 – 1:22.0 | Push Chair, unseen environment | What the Policies Learn |
| `zip_tie.mp4` | 1:22.4 – 1:39.0 | Zip-Tie Fastening, 1× | Policy Rollouts |
| `lego.mp4` | 1:39.4 – 2:04.7 | Disassemble Lego, 1× | Policy Rollouts |
| `lego_strategy.mp4` | 2:05.0 – 2:24.7 | Lego, same policy / another strategy, 1.5× | Policy Rollouts |
| `box_conveyor.mp4` | 2:25.0 – 3:04.6 | Box to Conveyor (long horizon), 3× | Policy Rollouts |
| `efficiency.mp4` | 3:05.0 – 3:37.9 | efficiency comparison vs VR / Mobile ALOHA, 4× | Collecting Demonstrations |
| `supplementary.mp4` | whole video (with audio) | full supplementary video | Supplementary Video |

Manual ffmpeg recipe for one clip (square pixels, 720p, muted, web-ready):

```sh
ffmpeg -ss 47.9 -to 60.0 -i master.mp4 -an \
  -vf "scale=trunc(iw*sar/2)*2:ih,setsar=1,scale=-2:720:flags=lanczos" \
  -c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p -movflags +faststart assets/video/push_chair.mp4
ffmpeg -ss 52 -i master.mp4 -frames:v 1 -vf "scale=-2:720" -q:v 3 assets/posters/push_chair.jpg
```

Guidelines from the template: H.264 MP4, 16:9, ideally under 10 MB per clip; autoplaying clips stay `muted playsinline`; clips below the fold load lazily through `data-src`.

### Using YouTube for the full video

The full video is 21 MB. To embed YouTube instead, replace the `<video>` inside `#video` with:

```html
<iframe src="https://www.youtube-nocookie.com/embed/VIDEO_ID" title="HALO supplementary video"
        allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen
        style="width:100%;height:100%;border:0"></iframe>
```

and delete `assets/video/supplementary.mp4` and its poster.

### Adding or changing a rollout in the demo switcher

Each `<button class="thumb">` in `#demos` has `data-src` (video), `data-poster`, `data-caption` (text under the player) and an `<img>` thumbnail. Duplicate a button to add a task; the first button must match the player's initial `data-src` / `poster`.

## Replacing the figures

`assets/figures/*.jpg` were cropped from the PDF at 300 dpi. For sharper images export them from your LaTeX/figure sources at about 1600 px wide and keep the file names:

| File | Paper figure | Used in |
|---|---|---|
| `fig1-interface.jpg` | Fig. 1 | How HALO Works |
| `fig3-end-effector.jpg` | Fig. 3 | Mechanical Design |
| `fig4-system.jpg` | Fig. 4 | Mechanical Design + 3D fallback image |
| `fig5-implicit-base.jpg` | Fig. 5 | Collecting Demonstrations |
| `fig6-visual-processing.jpg` | Fig. 6 | What the Policies Learn (side by side with Fig. 10; column widths in `.fig-row--6-10` follow the image aspect ratios) |
| `fig8-tasks.jpg` | Fig. 8 | What the Policies Learn |
| `fig9-long-horizon.jpg` | Fig. 9 | What the Policies Learn |
| `fig10-failure-cases.jpg` | Fig. 10 | What the Policies Learn |
| `fig11-generalization.jpg` | Fig. 11 | What the Policies Learn |

Update the `alt` text if a figure changes.

## Colours and key-word highlights

The palette follows the paper's figures (tokens at the top of `style.css`):

| Token | Value | From the paper |
|---|---|---|
| `--c-ours` | `#F89A42` | "Natural Locomotion" orange (Figs. 1, 2): active steps, buttons, ICR marker |
| `--c-accent` | `#A9530F` | the same orange darkened for text (5.3:1 on white): links, labels |
| `--c-series-1/2/3` | `#F8D45B` / `#388038` / `#A2BD79` | Fig. 7 bars: HALO / Mobile ALOHA / VR (charts draw the paper's dark bar shadow via `data-shadow`) |
| `--c-perception` | `#745BD0` | "Perception Gap" purple (Fig. 1) |
| `--c-mobility`, `--c-mobility-ink` | `#70AD47`, `#3F7A22` | "Mobility Gap" green (Fig. 1), and a text-safe shade |
| `--c-mark` | gold at 55% | marker behind highlighted key words |

The 3D labels, frustums and wheel markers use the Fig. 4 colour code (wrist cam `#2196F3`, head cam `#F89942`, drive wheel `#92D050`, encoder `#9D68F4`, fixed-direction wheel `#00B0F0`).

Highlight key words in the copy with:

- `<span class="kw">…</span>`: key claim or number (gold marker, semibold)
- `<span class="kw-loco">…</span>`: natural locomotion (paper orange)
- `<span class="kw-perception">…</span>` / `<span class="kw-mobility">…</span>`: perception gap / mobility gap
- `<span class="acr">…</span>`: the H-A-L-O letters in the subtitle

Keep highlights to a few per paragraph so they still stand out.

## Charts

Every chart is drawn by `page.js` from the `<table>` inside its `<figure class="chart">`. Edit the table, not the drawing code. `data-max` sets the axis top (axes start at 0), `data-unit` the suffix, and `data-min-width` the width below which the chart scrolls sideways on phones.

The two efficiency charts (demonstration time and success) use values **measured from the vector bars of Fig. 7**, because the paper does not print them: time 11 / 32 / 30 s (Push Chair) and 23 / 64 / 37 s (Lego); success 100 / 100 / 25 % and 100 / 80 / 20 % for HALO / Mobile ALOHA / VR. They agree with the text (2.9× and 2.8× speed-ups, up to 66% shorter, +20 pp). Replace them with your raw numbers if they differ. All other values are quoted from the paper's text and tables.

## The 3D viewer

### What the model is made of

| Part | Source |
|---|---|
| ARX5 arms (links 1–6), HALO long fingers, white finger-pad sleeves, rack-and-slider gripper, red wrist-camera mounts, wrist cameras, blue head-camera bracket and head camera | `robots/x5/` in the `aloha_sim2real` package (robot.urdf + meshes + robot.appearance.json colours) |
| AgileX Tracer 2.0 body, drive wheels, casters | AgileX [`ugv_gazebo_sim`](https://github.com/agilexrobotics/ugv_gazebo_sim), `tracer/tracer2_description` (commit 27633a9, package licence: BSD). The meshes have no colours; `build_tracer_bundle.py` colours them to match the photo. |
| Aluminium framework (880 × 710 mm), arm plates, head-camera mast, rear casters, fixed-direction wheels with encoders, operator | modelled in `halo3d.js` from the paper's dimensions and Fig. 4 |

The bundles are decimated (about 35k triangles per arm, 24k for the Tracer) and stored as quantized binary buffers (`x5_arm.bin` 0.55 MB, `tracer2.bin` 0.36 MB). Both arms share one set of meshes.

### Rebuilding the mesh bundles

Needs Python 3 with numpy and scipy.

```sh
# arms, gripper and cameras (path = robots/x5 folder of the aloha_sim2real package)
python3 tools/build_arm_bundle.py /path/to/lingbot_sim2real/assets/robots/x5 assets/models
# Tracer 2.0 (path = tracer2_description from agilexrobotics/ugv_gazebo_sim, branch master)
python3 tools/build_tracer_bundle.py /path/to/ugv_gazebo_sim/tracer/tracer2_description assets/models
```

Options: `--budget 0.5` halves the triangle budgets (lighter page), `--crease 38` sets the normal-smoothing angle. Per-mesh budgets are in `BUDGET` at the top of `build_arm_bundle.py`; Tracer colours are in `COL` in `build_tracer_bundle.py`. If you later get CAD meshes of the real HALO frame, export them as STL and add them to a bundle the same way, then remove the procedural frame block in `halo3d.js` (section "framework").

### Common edits in `halo3d.js`

| Change | Where |
|---|---|
| Frame dimensions (top-rail height 0.90 m, rail positions, mast) | constants `TOP`, `FX`, `FZ`, `BOTC`, `MIDC` under "framework" |
| Arm mount position / spacing | `ARM_X`, `ARM_Y`, `ARM_Z` (the URDF mount is ±0.3146 m) |
| Arm poses (rest = Fig. 4 pose, operator hold, close-up, deployment demo) | `POSE` (joint angles q1–q6 in rad + finger opening in m, solved on the URDF) |
| Callout labels per step | `LBL` (`s` = step index, `key: 1` = also shown on phones) |
| Camera framing per step | `preset()` |
| Hide the translucent operator | `SHOW_OPERATOR = false` |
| Step text | the `.mech-step` cards in `index.html` (`data-title` feeds the caption in the stage) |

After editing `style.css`, `page.js`, `halo3d.js` or `halo-models.js`, bump the `?v=` query where the file is loaded (`index.html`, and the `halo-models.js` import at the top of `halo3d.js`) so browsers drop the cached copy.

The viewer pauses when off-screen, respects `prefers-reduced-motion` (no auto-rotation, static poses, instant transitions), disables drag-to-rotate on touch screens so the page scrolls normally, and never captures the mouse wheel.

## Before publishing

- Authors, affiliations, links and BibTeX filled in; no `TODO` left in `index.html`.
- Preview clips replaced by full-resolution cuts (`tools/cut_clips.sh`).
- Checked at desktop and phone width through a local server: no sideways scrolling, videos play, the 3D model loads.
- If the paper is still under double-blind review, decide whether the page (and `assets/paper/HALO.pdf`) should be public yet.

## Credits and licences

- three.js r169, MIT licence (`vendor/three/LICENSE`).
- AgileX Tracer 2.0 meshes: agilexrobotics/ugv_gazebo_sim, BSD (per the package's `package.xml`).
- ARX5 arm link meshes: ARX official model release (ARXroboticsX/ARX_Model, X5/X5A), via the robot-description package; HALO gripper, sleeves and camera mounts are your own parts. Check ARX's terms before redistributing the meshes.
- Page template: XU Lab, HKUST.
