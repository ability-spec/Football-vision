# What completes the current MVP

The accepted scope is the bounded CPU workflow plus a learned player-candidate
detector: local football video plus manual
play boundaries produces reviewable predictions, observed play metrics, CSV,
JSON, preview images, an HTML report and a synchronized review video. This is a
research baseline. It does not yet establish accurate perception on real games.
The YOLOX-Tiny option uses official pretrained COCO person weights; integration
and actual model inference are implemented, while football-specific accuracy and
fine-tuning remain unvalidated. See [the model guide](LEARNED_DETECTOR.md).

## Completion criteria

| Criterion | Evidence / remaining work |
| --- | --- |
| Public command works without model downloads or GPU | Synthetic demo and CLI integration tests exercise the entire workflow. |
| Invalid inputs and missing evidence have explicit outcomes | Tests cover play conflicts, unknown geometry, coordinate discontinuities, camera cuts, missing observations and export failures. Keep unavailable metrics distinct from zero. |
| Results are inspectable and traceable | Video/source hashes, metric definitions, schema metadata, artifact hashes, HTML, PNG and AVI are exported. |
| A user can prepare independent evaluation inputs | Frame-preparation command creates unscaled source PNGs and pending labels; the scorer refuses incomplete templates. |
| Learned perception is integrated into the same workflow | Local YOLOX-Tiny ONNX weights run through OpenCV CPU; model identity/profile are recorded and no fallback or inference-time download occurs. |
| Results on representative real footage are measured | Four real NFL clips from three plays are available locally; full decoding and sampled learned inference passed. Independent player labels and held-out footage are still needed. No real-game detection/tracking accuracy claim is currently supported. |
| Detector quality satisfies a chosen use case | Requires user-agreed precision/recall, identity continuity and field-coverage goals. Measure the existing baseline first; choose learned perception scope using that evidence. |

Passing software tests completes integration checks, not the last two criteria.

## Inputs needed from the project owner

Initial real footage has been supplied: four Kaggle Helmet Assignment clips,
including both views of play `58102_002798`. All are development material. See
[the first review](NFL_INITIAL_REVIEW.md) for verified media properties and
observed detector limitations. Additional footage should prioritize unseen games
and the missing paired views; uploading clips does not complete independent labels.

Native helmet and sensor labels have also been supplied. The native reader and
reproducible comparison tool are implemented. Optional tiled inference improves
sampled helmet-center coverage from 247 to 364 matches, without changing the
person threshold. Both paired 366-frame clips complete the full workflow.
Independent whole-player accuracy is still unmeasured; tracking fragmentation
and unstable/unavailable field coordinates remain major product limitations.
See the footage review for exact evidence and refusals. Integration is working;
the real-video quality acceptance criteria are not complete.

1. Supply additional permitted clips from previously uninspected games for
   held-out testing after development choices are finalized. The initial four
   clips are available and are not a data-access blocker. Keep raw media outside
   Git. Include camera motion, occlusion and a cut where possible.
2. The owner selected the workflow plus learned detection. Confirm the required
   quality after inspecting measured results; decide whether football-specific
   fine-tuning is needed. Ball tracking and possession remain separate work.
3. Give manual start/snap/end boundaries for the selected plays. Independently
   label some extracted frames to score detection/tracking. Field-coordinate
   scoring additionally needs independent landmark alignment.
4. Agree on the acceptable output after inspecting the first measured examples.
   A completion target without a measurable quality requirement cannot establish
   that a vision model performs well enough for the intended use.

Routine code changes, bug fixes, local demos, tests, documentation and performance
investigation can proceed without another owner decision. Commits and pushes
remain owner-directed. Model choices, data rights and the final product scope
require owner input; no scope change is inferred from a passing synthetic demo.

## Local execution instruction

> Finish the bounded CPU Football-vision MVP locally: one command from video and
> manual play boundaries to a reviewable report, JSON/CSV, preview and video.
> Include learned detection with verified local weights and model provenance.
> Fix reproducible defects, preserve missing-evidence and coordinate contracts,
> test the actual public workflow, and document acceptance evidence. Use the
> provided clips and independent labels for real-video evaluation. Report exact
> remaining blockers and assumptions. Do not invent accuracy, commit or push.
> Keep completed changes ready for focused
> commits when requested.

This specifies a reviewable result. It does not promise completion of all original
roadmap phases, football-specific training, live streaming, ball/possession inference or
cross-sport generalization in a single coding pass.
