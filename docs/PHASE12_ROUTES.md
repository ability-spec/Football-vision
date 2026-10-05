# Phase 12: observed route and movement geometry

```python
from football_vision.analytics import analyze_route

result = analyze_route(
    samples, track_id=receiver_track_id,
    start_frame=play_start, end_frame=play_end, snap_frame=resolved_snap,
    offense_direction=1, defender_samples=annotated_defender_samples,
    crossing_y=26.0,
).to_dict()
```

The inclusive frame window and snap come from the caller, typically a resolved
Phase 10 segment. `offense_direction` is +1 or -1 along field x. Defender tracks
are caller-selected; the API never guesses receiver/defender roles from team IDs.
The optional lateral crossing reference is in the same coordinate system as the
samples. Teams, roles, LOS and absolute yard lines are not inferred.

Each continuous segment reports:

- Path length: sum of consecutive observed position distances in yards.
- Forward depth: maximum signed x displacement from that segment's first sample.
- Forward and lateral displacement: signed endpoint differences.
- Pre-snap distance: edges whose two endpoints occur strictly before snap.
- Net stem: longitudinal, lateral, mixed or stationary according to endpoint
  displacement. Defaults require 1 yard of displacement and a 2:1 dominance
  ratio. This describes net geometry, not initial route stem or a route name;
  a loop returning to its origin can have a stationary net stem and nonzero path
  length. A single observation has unknown stem.
- Crossing frames: first observation on the opposite side of the supplied y
  reference. Merely touching the line and returning is not a crossing. A zero
  plateau followed by a side change records the first opposite-side frame; the
  exact crossing time is not estimated.

Nearest defender distance uses only accepted measurements with the same frame,
coordinate frame, segment, mode and timestamp (within 1 microsecond). Missing
compatible defenders yield null. Ties resolve by track ID. This is visible-player
separation, not proof that the closest real defender was detected.

Predicted, coasted, rejected, nonfinite and unknown-coordinate positions cannot
contribute. Frame gaps, coordinate changes and non-increasing timestamps start
new segments. Distance and depth never bridge these boundaries. Each segment
carries coordinate provenance; relative modes are allowed for local geometry.
A result needs at least one consecutive measured edge to be available. Isolated
positions may still supply explicitly labeled per-frame separation.

No named routes, player-role recognition, uncertainty calibration or real-video
accuracy are claimed. Tests validate deterministic geometry, missing evidence,
coordinate boundaries, separation synchronization and crossing behavior. Caller
annotations must be updated when coordinate systems change.
