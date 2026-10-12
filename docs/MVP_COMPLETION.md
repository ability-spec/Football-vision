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
| Multiple clips can be reviewed reproducibly | The [batch review command](BATCH_REVIEW.md) accepts a relative-path manifest, preserves source/model provenance, and exports per-clip reviews plus one downloadable report. Manual play boundaries are still required for play analytics. |
| Results on representative real footage are measured | Twelve real NFL clips from seven games and seven plays are available locally. The additional eight clips contain 2,906 decoded frames and were reviewed with a fixed detector configuration. Native helmet-center diagnostics and a [four-frame manual player-box pilot](MANUAL_DETECTION_PILOT.md) are available; broader detection sampling, identity evaluation and final held-out footage are still needed. |
| Detector quality satisfies a chosen use case | Requires user-agreed precision/recall, identity continuity and field-coverage goals. Measure the existing baseline first; choose learned perception scope using that evidence. |

Passing software tests completes integration checks, not the last two criteria.

## Inputs needed from the project owner

Real footage has been supplied: twelve Kaggle Helmet Assignment clips from
seven distinct `gameKey` values and seven plays. The initial four cover three
games and three plays, including both views of `58102_002798`; the additional
eight cover four other games and four plays. All have now been inspected. See
[the first review](NFL_INITIAL_REVIEW.md) for verified media properties and
observed detector limitations and [the batch guide](BATCH_REVIEW.md) for the
reproducible multi-clip workflow. Additional footage should prioritize unseen
games; uploading clips does not complete independent labels.

Native helmet and sensor labels have also been supplied. The native reader and
reproducible comparison tool are implemented. Optional tiled inference improves
sampled helmet-center coverage from 247 to 364 matches, without changing the
person threshold. Both paired 366-frame clips complete the full workflow.
The owner-confirmed four-frame manual player-box pilot now measures sampled
precision/recall: baseline 79.2%/54.5%, tiled 71.2%/67.5%, at IoU >= 0.5.
This is development evidence, without a second-annotator audit or full-clip
accuracy claim. Tracking fragmentation and unstable/unavailable field
coordinates remain major product limitations.
The existing local review of the additional eight clips recorded 40,171
one-to-one helmet-center matches across 53,843 labeled on-field helmet
instances. These are repeated frame instances, not unique players or player
recall/AP. That earlier local report is separate from the public batch command.
Its fixed-configuration review does not preserve an untouched holdout for
later tuning: inspected clips become development material when their results
inform changes. Keep both views of each play together and split by `gameKey`.
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
