# Phase 11: observed pre-snap geometry

`football_vision.analytics.analyze_formation` summarizes a caller-selected frame
strictly before a resolved snap. It accepts trajectory samples and explicit team
assignments; it does not infer offense from team cluster labels.

```python
from football_vision.analytics import analyze_formation

snapshot = analyze_formation(
    samples, frame_id=presnap_frame, snap_frame=resolved_snap_frame,
    team_by_track=team_assignments,
    offense_team="home", defense_team="away",
    line_of_scrimmage_x=40.0, offense_direction=1, box_center_y=26.0,
)
result = snapshot.to_dict()
```

Pass samples from the trajectory builder. Use a resolved manual or estimated
Phase 10 snap; an unavailable snap cannot support this calculation. LOS and box
center are optional manual annotations in the samples' coordinate frame.

Metrics use yards: lateral width, longitudinal depth, centroid and mean nearest
teammate distance. Box count is the number of observed defenders within the
inclusive rectangle from LOS to five yards toward the defense, and five yards
either side of the supplied center. Its dimensions are configurable and exported.
Missing annotations or missing defensive observations produce null, not zero.

Only observed, accepted measured positions contribute. Predictions, coasted
tracks, unknown geometry, rejected and nonfinite positions are excluded. Mixed
coordinate frames, modes or segments refuse the snapshot. Relative coordinates
support local spacing but require LOS annotations in that same relative frame.
Unknown team assignments are listed separately. Counts describe visible players;
there is no claim that an incomplete camera view contains all 11 players.

This provides geometry, not named formations, player roles, ball placement or
statistically calibrated confidence. Tests cover deterministic geometry and
refusals; real-video formation accuracy remains unmeasured.
