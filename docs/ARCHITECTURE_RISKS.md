# Architecture risks in the CPU MVP

The current workflow is deliberately bounded (300 frames by default). These are
the main places where extending it can introduce errors or exceed its limits.

| Boundary | Failure risk | Current protection / next step |
| --- | --- | --- |
| Camera calibration → image-space tracker → trajectories | A cut can assign an old player's ID to a new player at a similar screen position. Resetting the ID allocator can also join unrelated trajectory histories. | Calibration now emits `camera_cut_detected` for both successful and failed new-shot calibration. The tracker retires active tracks without reusing IDs; the builder clears prediction state even for absent tracks. This deliberately does not identify the same athlete across shots. |
| Image coordinates → field coordinates → analytics | A changed origin or scale can look like player movement. | Coordinate-frame IDs and segments separate motion estimates. Keep these contracts when adding learned calibration; matrix updates alone must not define physical continuity. |
| Detection/team identity → play interpretation | An observed box does not establish a unique athlete, team, offense direction, possession, or event. | The CPU workflow leaves teams and offense unknown. Learned detection, jersey identity and ball possession need separate evidence and held-out evaluation before enabling dependent metrics. |
| Video decoding → motion timestamps | `frame_id / fps` assumes a constant frame rate. Variable-frame-rate footage can distort speed and acceleration. | The runner checks positive finite FPS, but does not ingest per-frame presentation timestamps. Normalize footage to constant FPS for this MVP; add a timestamp contract before supporting variable-rate input. |
| Frame loop → export → visualization | All frame records and all trajectory samples are retained; exported dictionaries add another representation. Full-game inputs can grow memory and JSON size substantially. | Keep the frame cap. Before lifting it, profile peak memory and introduce chunked/streaming artifacts with explicit play and shot boundaries. |
| Typed internal data → JSON/CSV/report | Multiple consumers and old saved artifacts can disagree about fields or metric meanings. | Exports carry schema/package/pipeline provenance. Add explicit migration or rejection rules when changing required fields or semantics; test real writer/reader pairs together. The cut flag is additive. |
| Decode → calibration → detection → rendering | A synchronous CPU pipeline may miss a future live-video latency target. Parallel workers can reorder frames or share mutable tracker state. | Current behavior is sequential and preserves frame order. Profile individual stages before adding workers; keep tracking/calibration state under one ordered owner. GPU throughput is unmeasured. |

## Limits of camera-cut protection

The cut detector uses histogram correlation, so it can miss visually similar
camera changes or flag lighting changes. The new lifecycle signal makes an
already detected cut safe across modules; it does not improve cut detection
accuracy. A successful calibration is not evidence that player IDs survived a
camera change. Geometry discontinuities caused only by recalibration are not
automatically player-identity resets.

The regression suite covers successful/failed calibration at a cut, matching
screen boxes belonging to separate shots, no detections on the cut frame,
absent-track prediction invalidation, a non-repeating cut flag, and video-runner
export. These synthetic contracts do not measure real-game accuracy.
