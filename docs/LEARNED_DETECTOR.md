# Pretrained player-candidate detector

The MVP supports official YOLOX-Tiny ONNX weights through OpenCV DNN on CPU.
This is a learned COCO **person** detector. It has not been fine-tuned for American
football and does not distinguish athletes, referees, coaches or spectators.
Team, offense direction and possession remain unknown. Real-game detection and
tracking accuracy must be measured using independently annotated clips.

## Install the local weights

No new Python inference dependency is required. Download the official model once:

```bash
mkdir -p models
curl --fail --location https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx --output models/yolox_tiny.onnx
```

On Windows, download that URL in a browser into `models/yolox_tiny.onnx`.
The supported file is 20,219,662 bytes with SHA-256:

```text
427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7
```

The constructor verifies this hash before loading the network. ONNX files are
ignored by Git. Inference never downloads a model and never falls back to a
different detector when weights are missing, incompatible or corrupt. The
adapter intentionally supports this one verified profile. Arbitrary ONNX exports
or fine-tuned weights require their own verified input/output profile.

## Run the same workflow

```bash
python -m football_vision clip.mp4 --plays plays.json --source-kind real --detector yolox --weights models/yolox_tiny.onnx --max-frames 300 --out outputs/learned-run
```

The prediction-only runner accepts the same selection:

```bash
python -m football_vision.evaluation.video clip.mp4 --source-kind real --detector yolox --weights models/yolox_tiny.onnx --out predictions.json
```

The default `--detector turf` remains the untrained CPU baseline and needs no
weights. YOLOX uses fixed 416×416 input, confidence 0.3 (objectness × person score)
and NMS IoU 0.45. Detection boxes are decoded and mapped back to original pixel
coordinates before footpoint estimation, tracking and field projection. Fully
padded or degenerate boxes are excluded; image-edge boxes are clipped. Detection
confidence does not certify that a person is a football player or that their
field projection is valid.

### Small-player mode

Use `--detector yolox-tiled` with the same verified weights in either command.
This runs the full image and four overlapping crops (a 2-by-2 layout with 20%
overlap), maps boxes to original image coordinates, discards fragments touching
internal crop boundaries and merges duplicates with global NMS. Footpoints are
recomputed in full-image coordinates. Images no larger than 416 pixels on either
axis use only the full-image pass. Model input, confidence and NMS thresholds
are unchanged. This costs up to five inference passes, and does not constitute
football-specific training or establish athlete/referee identity.

The CLIs set `--opencv-threads 2` by default; increase or lower this positive
thread budget for the machine. Library calls preserve the caller's OpenCV thread
configuration. Detector metadata records the observed OpenCV thread budget.

On six sampled frames from each of four owner-supplied development clips,
one-to-one containment matches with native on-field helmet centers increased
from 247 to 364 out of 468 labeled helmets. This is a diagnostic, not recall,
precision or AP: large/wrong person boxes can contain centers, and false
positives are not scored. Neither player-box accuracy nor held-out generalization
has been established. See [the footage review](NFL_INITIAL_REVIEW.md).

Results identify the selected detector/source, settings, model hash, source URL,
COCO person class and preprocessing profile. The report names the selected
detector. `detector_accuracy_status` remains `real_video_unmeasured` for both
detectors until an independent evaluation exists. Keep evaluation outputs separate
from prediction artifacts; loading pretrained weights is not an accuracy result.

## Reference review: an incompatible example

The tagged [0.1.1rc0 demo](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.1.1rc0/demo/ONNXRuntime/onnx_inference.py)
uses RGB/ImageNet normalization via that tag's `preproc`. It is incompatible with
the currently published `yolox_tiny.onnx` artifact above. On the upstream
`assets/dog.jpg`, that preparation gave a maximum objectness × class score of
about 0.030. Unnormalized BGR produced about 0.887 and nonzero bicycle/vehicle/animal
scores. These are a compatibility smoke check, not a quantitative accuracy benchmark.

The adapter therefore uses top-left padding with pixel value 114, unnormalized
BGR values in [0,255], and contiguous float32 NCHW input. This matches the
[0.3.0 preprocessing](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/yolox/data/data_augment.py).
The release has raw stride-8/16/32 grid outputs with 80 COCO classes. A checksum
binds these assumptions to the actual artifact rather than the URL's version tag.
Wrong preprocessing can produce finite output and apparently successful empty
detections, so network loading alone is an insufficient validation check.

The upstream YOLOX repository uses [Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/LICENSE).
Weights remain a separately downloaded upstream artifact; the project does not
relicense or redistribute them under its MIT license.

## Validation and remaining work

Contract tests cover input scale/channel order, raw-grid decoding, original-size
coordinates, person-score selection, NMS, padding, incompatible outputs, model
identity, and the complete report workflow. They use a controlled network output;
they do not measure learned detector quality. A local optional smoke test runs the
actual official model; CI does not download weights. A synthetic-video run with
the actual model verifies the public workflow but is not a football benchmark.

Next, compare turf and YOLOX on the same independently labeled development clips,
inspect misses and sideline false positives, and select parameters on development
data. Only then evaluate held-out games/cameras. Football fine-tuning, if needed
to meet agreed targets, requires player annotations and separate training and
validation data; it is not represented as complete by this pretrained adapter.
