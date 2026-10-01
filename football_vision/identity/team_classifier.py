"""Baseline torso appearance TeamClassifier with deterministic labels and UNKNOWN abstention (Phase 3D).

Uses CIE Lab + HSV torso crop features with turf-background masking and
deterministic 2-centroid clustering:
  - ``TEAM_A``: darker / lower-luminance jersey cluster centroid
  - ``TEAM_B``: lighter / higher-luminance jersey cluster centroid
  - ``UNKNOWN``: ambiguous margin, insufficient non-turf torso pixels, or
    extreme appearance outlier (e.g., striped official or sideline vest)

Never hardcodes team RGB colors for a single game.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from football_vision.schema import PlayerDetection, TeamAssignment, TeamLabel


class TorsoTeamClassifier:
    """Unsupervised, deterministic torso appearance classifier for football broadcasts."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.45,
        min_non_turf_ratio: float = 0.18,
        min_margin_ratio: float = 0.14,
        max_cluster_dist: float = 75.0,
        min_centroid_separation: float = 14.0,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_non_turf_ratio = float(min_non_turf_ratio)
        self.min_margin_ratio = float(min_margin_ratio)
        self.max_cluster_dist = float(max_cluster_dist)
        self.min_centroid_separation = float(min_centroid_separation)
        self._centroid_a: Optional[np.ndarray] = None  # Darker jersey (lower L)
        self._centroid_b: Optional[np.ndarray] = None  # Lighter jersey (higher L)

    def reset(self) -> None:
        """Reset learned session centroids."""
        self._centroid_a = None
        self._centroid_b = None

    @property
    def centroids(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._centroid_a is None or self._centroid_b is None:
            return None
        return (self._centroid_a.copy(), self._centroid_b.copy())

    def extract_torso_features(
        self,
        frame_bgr: np.ndarray,
        bbox: Sequence[float],
    ) -> Dict[str, Any]:
        """Extract turf-masked CIE Lab + HSV torso features from a player bounding box."""
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Torso region: upper-middle 16%..54% vertically, central 20%..80% horizontally
        tx1 = int(np.clip(np.floor(x1 + 0.20 * bw), 0, w_img - 1))
        tx2 = int(np.clip(np.ceil(x2 - 0.20 * bw), tx1 + 1, w_img))
        ty1 = int(np.clip(np.floor(y1 + 0.16 * bh), 0, h_img - 1))
        ty2 = int(np.clip(np.ceil(y1 + 0.54 * bh), ty1 + 1, h_img))

        patch_bgr = frame_bgr[ty1:ty2, tx1:tx2]
        if patch_bgr.size == 0:
            return {
                "valid": False,
                "feature": np.zeros(4, dtype=np.float64),
                "non_turf_ratio": 0.0,
                "luma": 0.0,
                "chroma": 0.0,
                "is_striped_referee": False,
            }

        patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)

        # Mask out green turf background pixels
        turf_mask = cv2.inRange(patch_hsv, (30, 40, 35), (86, 255, 255)) > 0
        fg_mask = ~turf_mask
        non_turf_ratio = float(np.mean(fg_mask))

        if np.sum(fg_mask) < 4:
            return {
                "valid": False,
                "feature": np.zeros(4, dtype=np.float64),
                "non_turf_ratio": non_turf_ratio,
                "luma": 0.0,
                "chroma": 0.0,
                "is_striped_referee": False,
            }

        L_vals = patch_lab[:, :, 0][fg_mask].astype(np.float64)
        a_vals = patch_lab[:, :, 1][fg_mask].astype(np.float64)
        b_vals = patch_lab[:, :, 2][fg_mask].astype(np.float64)
        S_vals = patch_hsv[:, :, 1][fg_mask].astype(np.float64)

        med_L = float(np.median(L_vals))
        med_a = float(np.median(a_vals))
        med_b = float(np.median(b_vals))
        med_S = float(np.median(S_vals))
        chroma = float(np.hypot(med_a - 128.0, med_b - 128.0))

        # Detect high-contrast black-and-white vertical referee stripes:
        # Low saturation + extreme bimodal luminance variance across columns
        col_luma = np.mean(patch_lab[:, :, 0].astype(np.float64), axis=0)
        col_diff = float(np.mean(np.abs(np.diff(col_luma)))) if len(col_luma) >= 4 else 0.0
        luma_std = float(np.std(L_vals))
        is_striped_referee = bool(med_S < 35.0 and luma_std > 62.0 and col_diff > 28.0)

        feat = np.array([med_L, med_a, med_b, 0.5 * med_S], dtype=np.float64)
        return {
            "valid": True,
            "feature": feat,
            "non_turf_ratio": round(non_turf_ratio, 4),
            "luma": round(med_L, 3),
            "chroma": round(chroma, 3),
            "is_striped_referee": is_striped_referee,
        }

    def _fit_two_centroids_deterministic(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic 2-centroid Lloyd clustering initialized by L-channel quantiles."""
        n = len(features)
        if n == 1:
            return features[0].copy(), features[0].copy()

        lumas = features[:, 0]
        idx_min = int(np.argmin(lumas))
        idx_max = int(np.argmax(lumas))
        if idx_min == idx_max:
            # Break ties using second feature component
            idx_min = 0
            idx_max = n - 1

        c_a = features[idx_min].copy()
        c_b = features[idx_max].copy()

        for _ in range(12):
            d_a = np.linalg.norm(features - c_a[None, :], axis=1)
            d_b = np.linalg.norm(features - c_b[None, :], axis=1)
            mask_a = d_a <= d_b
            if np.all(mask_a) or not np.any(mask_a):
                break
            new_a = np.median(features[mask_a], axis=0)
            new_b = np.median(features[~mask_a], axis=0)
            if np.allclose(new_a, c_a, atol=1e-3) and np.allclose(new_b, c_b, atol=1e-3):
                c_a, c_b = new_a, new_b
                break
            c_a, c_b = new_a, new_b

        # Enforce deterministic canonical ordering:
        # TEAM_A has strictly lower luminance L (or lower a-channel on exact L tie)
        if (c_a[0] > c_b[0] + 1e-4) or (abs(c_a[0] - c_b[0]) <= 1e-4 and c_a[1] > c_b[1]):
            c_a, c_b = c_b, c_a

        return c_a, c_b

    def classify_detections(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[PlayerDetection],
        *,
        update_centroids: bool = True,
    ) -> List[TeamAssignment]:
        """Assign each detection to ``TEAM_A``, ``TEAM_B``, or ``UNKNOWN``."""
        if not detections:
            return []

        extracted = [self.extract_torso_features(frame_bgr, det.bbox) for det in detections]
        valid_feats = [
            e["feature"]
            for e in extracted
            if e["valid"]
            and e["non_turf_ratio"] >= self.min_non_turf_ratio
            and not e["is_striped_referee"]
        ]

        if len(valid_feats) >= 2 and update_centroids:
            feat_mat = np.vstack(valid_feats)
            obs_a, obs_b = self._fit_two_centroids_deterministic(feat_mat)
            sep = float(np.linalg.norm(obs_a - obs_b))
            if sep >= self.min_centroid_separation:
                if self._centroid_a is None or self._centroid_b is None:
                    self._centroid_a, self._centroid_b = obs_a, obs_b
                else:
                    # Smooth centroid update preserving TEAM_A (darker) / TEAM_B (lighter)
                    self._centroid_a = 0.75 * self._centroid_a + 0.25 * obs_a
                    self._centroid_b = 0.75 * self._centroid_b + 0.25 * obs_b
                    if self._centroid_a[0] > self._centroid_b[0]:
                        self._centroid_a, self._centroid_b = self._centroid_b, self._centroid_a

        assignments: List[TeamAssignment] = []
        for ext in extracted:
            if not ext["valid"] or ext["non_turf_ratio"] < self.min_non_turf_ratio:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=0.0,
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="insufficient_non_turf_torso_pixels",
                    )
                )
                continue

            if ext["is_striped_referee"]:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=0.0,
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="striped_official_pattern",
                    )
                )
                continue

            if self._centroid_a is None or self._centroid_b is None:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=0.0,
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="uninitialized_team_centroids",
                    )
                )
                continue

            feat = ext["feature"]
            d_a = float(np.linalg.norm(feat - self._centroid_a))
            d_b = float(np.linalg.norm(feat - self._centroid_b))
            d_min = min(d_a, d_b)
            d_sum = max(1e-6, d_a + d_b)
            margin = abs(d_b - d_a) / d_sum

            if margin < self.min_margin_ratio:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=round(float(margin / self.min_margin_ratio) * 0.40, 4),
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="ambiguous_cluster_margin",
                    )
                )
                continue

            if d_min > self.max_cluster_dist:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=round(float(max(0.0, 0.30 * (1.0 - d_min / 150.0))), 4),
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="appearance_outlier_distance",
                    )
                )
                continue

            # Compute confidence from cluster separation margin, proximity, and torso fill
            prox_score = float(np.clip(1.0 - d_min / self.max_cluster_dist, 0.0, 1.0))
            margin_score = float(np.clip(margin / 0.65, 0.0, 1.0))
            fill_score = float(np.clip(ext["non_turf_ratio"] / 0.65, 0.35, 1.0))
            conf = float(np.clip(0.45 * margin_score + 0.35 * prox_score + 0.20 * fill_score, 0.0, 0.99))

            if conf < self.min_confidence:
                assignments.append(
                    TeamAssignment(
                        team="UNKNOWN",
                        confidence=round(conf, 4),
                        torso_luma=ext["luma"],
                        torso_chroma=ext["chroma"],
                        non_turf_ratio=ext["non_turf_ratio"],
                        reason="low_team_confidence",
                    )
                )
                continue

            label: TeamLabel = "TEAM_A" if d_a < d_b else "TEAM_B"
            assignments.append(
                TeamAssignment(
                    team=label,
                    confidence=round(conf, 4),
                    torso_luma=ext["luma"],
                    torso_chroma=ext["chroma"],
                    non_turf_ratio=ext["non_turf_ratio"],
                    reason=None,
                )
            )

        return assignments
