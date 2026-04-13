"""Scoring framework for the synthetic benchmark suite.

Compares detected events against the ground-truth manifest using
temporal IoU matching and the Hungarian algorithm for optimal
one-to-one assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from qp.benchmark.manifest import DatasetManifest
from qp.events.catalog import WavePacketPeak


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One matched pair: ground truth event ↔ detection."""

    gt_event_id: str
    detected_idx: int | None  # None if GT was missed
    iou: float
    period_error_pct: float
    band_correct: bool


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """Aggregate score for one dataset."""

    dataset_id: str
    n_ground_truth: int
    n_detectable: int
    n_detected: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    precision: float
    recall: float
    f1: float
    band_accuracy: float
    mean_period_error_pct: float
    mean_iou: float
    recall_by_difficulty: dict[str, float]
    n_decoy_events: int
    n_decoy_detected: int
    decoy_rejection_rate: float
    matches: list[MatchResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SuiteScore:
    """Aggregate score across all benchmark datasets."""

    overall_precision: float
    overall_recall: float
    overall_f1: float
    per_tier_recall: dict[str, float]
    per_band_recall: dict[str, float]
    band_accuracy: float
    decoy_rejection_rate: float
    summary_score: float  # harmonic mean of (F1, band_accuracy, decoy_rejection)
    dataset_scores: list[BenchmarkScore] = field(default_factory=list)


def _time_iou(
    gt_start: float, gt_end: float,
    det_start: float, det_end: float,
) -> float:
    """Intersection over union of two time intervals."""
    inter_start = max(gt_start, det_start)
    inter_end = min(gt_end, det_end)
    intersection = max(0.0, inter_end - inter_start)
    union = max(gt_end - gt_start, 0) + max(det_end - det_start, 0) - intersection
    return intersection / union if union > 0 else 0.0


def _det_time_range(det: WavePacketPeak, t0_sec: float = 0.0) -> tuple[float, float]:
    """Extract start/end times (seconds) from a detection."""
    from_sec = det.date_from.timestamp()
    to_sec = det.date_to.timestamp()
    return from_sec - t0_sec, to_sec - t0_sec


def score_dataset(
    manifest: DatasetManifest,
    detections: list[WavePacketPeak],
    iou_threshold: float = 0.2,
    t0_sec: float = 0.0,
) -> BenchmarkScore:
    r"""Score pipeline detections against ground truth.

    Parameters
    ----------
    manifest : DatasetManifest
        Ground truth with injected events.
    detections : list[WavePacketPeak]
        Pipeline output.
    iou_threshold : float
        Minimum IoU to consider a match.
    t0_sec : float
        Epoch offset (seconds) to align detection timestamps with
        the manifest's time axis. Typically 0 for synthetic data
        where detections use datetime(2000, 1, 1) + timedelta(seconds=t).
    """
    gt_detectable = [e for e in manifest.events if e.should_detect]
    gt_decoys = [e for e in manifest.events if not e.should_detect]
    n_gt = len(gt_detectable)
    n_det = len(detections)

    # Build IoU cost matrix (gt × detections)
    if n_gt > 0 and n_det > 0:
        iou_matrix = np.zeros((n_gt, n_det))
        for i, gt in enumerate(gt_detectable):
            for j, det in enumerate(detections):
                det_start, det_end = _det_time_range(det, t0_sec)
                iou_matrix[i, j] = _time_iou(
                    gt.start_sec, gt.end_sec, det_start, det_end
                )

        # Hungarian assignment (maximise IoU → minimise negative IoU)
        cost = -iou_matrix
        gt_idx, det_idx = linear_sum_assignment(cost)

        matches: list[MatchResult] = []
        matched_det: set[int] = set()
        matched_gt: set[int] = set()

        for gi, di in zip(gt_idx, det_idx):
            iou = iou_matrix[gi, di]
            if iou >= iou_threshold:
                gt_ev = gt_detectable[gi]
                det_ev = detections[di]
                det_period = det_ev.period_sec or 0.0
                period_err = (
                    abs(det_period - gt_ev.period_sec) / gt_ev.period_sec * 100
                    if gt_ev.period_sec > 0
                    else 0.0
                )
                band_ok = det_ev.band == gt_ev.band if gt_ev.band else True
                matches.append(MatchResult(
                    gt_event_id=gt_ev.event_id,
                    detected_idx=di,
                    iou=iou,
                    period_error_pct=period_err,
                    band_correct=band_ok,
                ))
                matched_det.add(di)
                matched_gt.add(gi)
    else:
        matches = []
        matched_det = set()
        matched_gt = set()

    tp = len(matches)
    fp = n_det - len(matched_det)
    fn = n_gt - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    band_acc = (
        sum(1 for m in matches if m.band_correct) / tp if tp > 0 else 1.0
    )
    mean_period_err = (
        float(np.mean([m.period_error_pct for m in matches])) if matches else 0.0
    )
    mean_iou = float(np.mean([m.iou for m in matches])) if matches else 0.0

    # Per-difficulty recall
    recall_by_diff: dict[str, float] = {}
    for diff in ("easy", "moderate", "hard", "extreme"):
        gt_diff = [e for e in gt_detectable if e.difficulty == diff]
        if not gt_diff:
            continue
        matched_diff = sum(
            1 for gi in matched_gt if gt_detectable[gi].difficulty == diff
        )
        recall_by_diff[diff] = matched_diff / len(gt_diff)

    # Decoy rejection
    n_decoy = len(gt_decoys)
    decoy_detected = 0
    for det in detections:
        det_start, det_end = _det_time_range(det, t0_sec)
        for decoy in gt_decoys:
            if _time_iou(decoy.start_sec, decoy.end_sec, det_start, det_end) > 0.1:
                decoy_detected += 1
                break
    decoy_rejection = 1.0 - decoy_detected / n_decoy if n_decoy > 0 else 1.0

    return BenchmarkScore(
        dataset_id=manifest.dataset_id,
        n_ground_truth=n_gt,
        n_detectable=manifest.n_detectable,
        n_detected=n_det,
        n_true_positives=tp,
        n_false_positives=fp,
        n_false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        band_accuracy=band_acc,
        mean_period_error_pct=mean_period_err,
        mean_iou=mean_iou,
        recall_by_difficulty=recall_by_diff,
        n_decoy_events=n_decoy,
        n_decoy_detected=decoy_detected,
        decoy_rejection_rate=decoy_rejection,
        matches=matches,
    )


def score_suite(scores: list[BenchmarkScore]) -> SuiteScore:
    """Aggregate scores across the entire benchmark suite."""
    total_tp = sum(s.n_true_positives for s in scores)
    total_fp = sum(s.n_false_positives for s in scores)
    total_fn = sum(s.n_false_negatives for s in scores)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    # Per-tier recall
    per_tier: dict[str, float] = {}
    for tier in ("tier1", "tier2", "tier3", "tier4", "decoy"):
        tier_scores = [
            s for s in scores if s.dataset_id.startswith(tier)
        ]
        if not tier_scores:
            continue
        tier_tp = sum(s.n_true_positives for s in tier_scores)
        tier_gt = sum(s.n_detectable for s in tier_scores)
        per_tier[tier] = tier_tp / tier_gt if tier_gt > 0 else 0.0

    per_band: dict[str, float] = {}  # TODO: aggregate GT band info

    # Band accuracy
    total_matched = sum(len(s.matches) for s in scores)
    band_acc = (
        sum(sum(1 for m in s.matches if m.band_correct) for s in scores)
        / total_matched
        if total_matched > 0
        else 1.0
    )

    # Decoy rejection
    total_decoy = sum(s.n_decoy_events for s in scores)
    total_decoy_det = sum(s.n_decoy_detected for s in scores)
    decoy_rej = 1.0 - total_decoy_det / total_decoy if total_decoy > 0 else 1.0

    # Summary: harmonic mean of F1, band_accuracy, decoy_rejection
    components = [x for x in (f1, band_acc, decoy_rej) if x > 0]
    summary = (
        len(components) / sum(1.0 / c for c in components)
        if components
        else 0.0
    )

    return SuiteScore(
        overall_precision=precision,
        overall_recall=recall,
        overall_f1=f1,
        per_tier_recall=per_tier,
        per_band_recall=per_band,
        band_accuracy=band_acc,
        decoy_rejection_rate=decoy_rej,
        summary_score=summary,
        dataset_scores=scores,
    )
