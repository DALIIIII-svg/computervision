# Kombat Vision

A Mortal Kombat–style fighting game controlled by **hand gestures** seen through a webcam,
built for the SUP'COM **Computer Vision** course.

The player fights a bot. Instead of a keyboard, the camera reads the player's hand shape
(thumb up = punch, four fingers = kick, open palm = block…) and head movement (head down = dodge).

The goal is not just a working game: the **perception pipeline is built with classical computer
vision**, then extended with learned classifiers (HOG + SVM, Bag of Features + SVM, CNN) and
compared against a deep-learning baseline (MediaPipe Hands).

---

## Project status

> **Phase 2 of 5: classical pipeline written, now being tuned on a real camera.**

| Part | Status | Notes |
|---|---|---|
| Game (fighters, bot, rounds, HUD) | ✅ Done | Fully playable with the keyboard |
| Input abstraction (keyboard / vision / MediaPipe) | ✅ Done | Same `Actions` object for every source |
| Classical vision pipeline v1 | ✅ Written | 26 automated tests pass on synthetic images |
| Head dodge (Viola-Jones + CamShift + Kalman) | ✅ Works on a real camera | |
| Hand gestures on a real camera | 🔧 **In progress** | Pipeline runs, but the hand is not yet detected reliably in our room; tuning needed (see below) |
| MediaPipe baseline | 🟡 Written, not yet tested on a real camera | |
| Dataset capture tool | ⏳ Next | |
| HOG + SVM / BoF + SVM / CNN classifiers | ⏳ Planned | |
| Evaluation and report | ⏳ Planned | |

### Current issue: hand gestures not detected

The dodge works because it only depends on face detection and tracking. A hand gesture needs
**three stages to succeed together**: background subtraction, skin segmentation and hand
extraction. If any one fails, no hand is found. Suspects, in order:

1. The player was in the frame while the background was being learned, so the player became "background".
2. Skin calibration sampled the wrong pixels (hand not fully in the yellow box).
3. Lighting or a skin-colored wall pollutes the skin mask.

**How to debug:** run `python scripts/vision_lab.py` and check the four panels
(camera, foreground, skin, hand). The first panel that looks wrong is the stage to tune in
`configs/default.yaml`. Try `--background none` to rule out background subtraction.

---

## Gestures

| Gesture | Action |
|---|---|
| Thumb up (one finger) | Punch |
| Four fingers | Kick |
| Open palm | Block (held) |
| Two fingers (peace) | Special move (costs energy) |
| Fist | Idle |
| Hand in the left / right third of the frame | Move left / right |
| Head drops quickly | Dodge (avoids punches and specials, not kicks) |

The mapping is editable in `configs/default.yaml` (`gestures.mapping`).

---

## Setup

Python 3.10 or newer (tested up to 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -r requirements-baseline.txt   # optional, for the MediaPipe baseline
```

Dependency notes:
- **pygame-ce** (community edition of pygame, same `import pygame`) is used because the original
  `pygame` has no prebuilt package for Python 3.14 and fails to compile on Windows.
- **OpenCV is pinned below 5.0**: OpenCV 5 removed the Haar cascade module used by Viola-Jones.
  The `contrib` build is used everywhere because MediaPipe depends on it; mixing two OpenCV
  packages in one environment breaks the install.

## Run

```bash
python main.py                              # keyboard only: test the game
python main.py --input vision --debug       # our pipeline + debug window
python main.py --input vision --background codebook
python main.py --input mediapipe            # deep-learning baseline
python scripts/vision_lab.py                # pipeline alone, for tuning (press R to record a video)
python main.py --input vision --video data/raw/session_xxx.mp4 --debug   # replay a recording
pytest                                      # run the tests
```

**Startup with the vision input**
1. **Step completely out of the frame** (~2 s) while the empty background is learned.
2. **Hold your open hand inside the yellow box** (~1.5 s) to calibrate your skin color.
3. Fight.

Keys: `B` relearn background · `C` recalibrate skin · `N` skip calibration ·
`R` restart round · `ESC` quit. Keyboard controls (`A/D` move, `J` punch, `K` kick, `L` block,
`U` special, `S` dodge) keep working alongside the camera.

Tips: sit 1–1.5 m from the camera, raise the hand **beside** your face (not in front of it)
with fingers up, avoid skin-colored backgrounds, prefer long sleeves.

---

## The perception pipeline

A perception pipeline turns raw camera frames into something the game understands. Each stage
takes the output of the previous one; the game never sees pixels, only actions like `PUNCH`.

```
frame ─► preprocess ─► background subtraction ─► foreground mask
                    ├► Viola-Jones + CamShift + Kalman (head) ─► face box ─► dodge
                    └► skin segmentation ─► AND foreground, minus face
                         ─► connected components ─► region growing ─► Canny split ─► wrist cut
                         ─► hand contour ─► CamShift + Kalman (hand position ─► move)
                         ─► convex hull + convexity defects ─► finger count ─► gesture
                         ─► stabilizer (N frames) ─► action mapping ─► Actions ─► game
```

| Stage | Technique | Why |
|---|---|---|
| Preprocessing | resize, Gaussian blur, YCrCb | speed, noise reduction, separate brightness from color |
| Background | **MOG2** (OpenCV) / **Codebook** (our NumPy implementation of Kim et al. 2005) | keep only the player; the two are compared under lighting changes |
| Skin | YCrCb thresholds, calibrated per user | skin is compact in (Cr, Cb); brightness is ignored |
| Face | **Viola-Jones** (Haar cascade) | remove the face from the skin mask; start the head tracker |
| Hand blob | connected components, **region growing** | pick the hand, recover pixels the threshold missed |
| Contour | **Canny** | cut along edges between touching fingers |
| Wrist cut | distance transform (palm circle) | stop the forearm from counting as a finger |
| Tracking | **CamShift (Mean Shift)** + **Kalman filter** (written out in NumPy) | follow hand/head when detection fails, estimate velocity |
| Gesture | **convex hull + convexity defects** | count fingers from the valleys between them, no training data |
| Time | stabilizer + cooldowns | ignore transition shapes, fire each attack once |

Design choices:
- The game runs at 60 FPS in the main thread; the camera and the pipeline run in their own
  threads, so slow vision never freezes the game.
- Every input source produces the same `Actions` object, so our pipeline, MediaPipe and the
  keyboard can be swapped and compared fairly.
- Every parameter lives in `configs/default.yaml`; the debug view shows each stage and its latency.

Known limitation of v1: convexity defects cannot tell a thumb from an index finger, so
"thumb up" and "one finger" are the same gesture. The learned classifiers will fix this.

---

## Architecture

```
computervision/
├── main.py                       entry point (input source, config, options)
├── configs/default.yaml          every tunable parameter, commented
├── kombat_vision/
│   ├── actions.py                Actions + Gesture: the contract between inputs and game
│   ├── config.py                 typed config (dataclasses) + YAML loading
│   ├── vision/                   ── perception pipeline ──
│   │   ├── camera.py             threaded capture (webcam or video file)
│   │   ├── preprocessing.py      resize, blur, YCrCb
│   │   ├── background/           base, MOG2, Codebook
│   │   ├── skin.py               skin model + calibration
│   │   ├── face.py               Viola-Jones
│   │   ├── hand.py               components, region growing, Canny split, wrist cut
│   │   ├── tracking.py           Kalman filter + CamShift tracker
│   │   ├── fingers.py            convex hull + convexity defects -> gesture
│   │   ├── dodge.py              head velocity / drop -> dodge
│   │   ├── temporal.py           stabilizer + gesture -> action mapping
│   │   ├── pipeline.py           orchestration, per-stage timing
│   │   └── debug_view.py         all stages side by side
│   ├── inputs/                   ── interchangeable input sources ──
│   │   ├── base.py, camera_input.py
│   │   ├── keyboard.py
│   │   ├── vision.py             our pipeline
│   │   └── mediapipe_baseline.py MediaPipe HandLandmarker + the same gesture logic
│   ├── game/                     ── the game ──
│   │   ├── fighter.py            states, attack frames, damage, block, dodge
│   │   ├── bot.py                finite-state-machine opponent with reaction time
│   │   ├── renderer.py           procedural drawing (no copyrighted sprites)
│   │   └── engine.py             60 FPS loop, collisions, rounds
│   └── utils/timing.py           latency and FPS measurement
├── scripts/vision_lab.py         tune the pipeline without the game, record videos
├── tests/                        pytest on synthetic hands (no camera needed)
├── data/                         recordings and datasets (git-ignored)
├── assets/                       free sprites, if added later
└── docs/                         report, figures, results
```

---

## Roadmap

**Phase 1 – Game shell** ✅
- [x] Fighters, bot, rounds, HUD, keyboard input

**Phase 2 – Classical pipeline (no training data)** 🔧 current
- [x] Background subtraction (MOG2, Codebook), skin calibration, Viola-Jones, region growing,
      Canny, convexity defects, CamShift + Kalman, dodge, debug view
- [x] MediaPipe baseline (code)
- [ ] Tune on a real camera until every gesture works in our room
- [ ] Test the MediaPipe baseline on a real camera

**Phase 3 – Data**
- [ ] Capture tool: hold a gesture, the pipeline saves auto-labeled hand crops
- [ ] Our dataset: ~6 gestures × 2 people × 2 lighting setups
- [ ] Optional pre-training on [HaGRID](https://github.com/hukenovs/hagrid) (public hand-gesture dataset)

**Phase 4 – Learned classifiers** (each one replaces `fingers.py`)
- [ ] HOG + SVM
- [ ] SIFT Bag of Features + SVM
- [ ] Small CNN / fine-tuned MobileNet

**Phase 5 – Evaluation and report**
- [ ] Per-gesture precision / recall, confusion matrices
- [ ] Latency per stage
- [ ] Ablations: MOG2 vs Codebook, with / without Kalman, convexity defects vs HOG vs BoF vs CNN vs MediaPipe
- [ ] Robustness: lighting, background, distance, skin tones
- [ ] Report, slides, live demo

---

## Credits

- The idea of separating the input layer from the game was inspired by
  [PlayFit-AI](https://github.com/i-am-jillian/PlayFit-AI) (MIT). No code or assets were copied.
- Codebook: K. Kim, T. Chalidabhongse, D. Harwood, L. Davis, *Real-time foreground–background
  segmentation using codebook model*, Real-Time Imaging, 2005.
- MOG2: Z. Zivkovic, *Improved adaptive Gaussian mixture model for background subtraction*, ICPR 2004.
- Viola-Jones: P. Viola, M. Jones, *Rapid object detection using a boosted cascade of simple features*, CVPR 2001.
