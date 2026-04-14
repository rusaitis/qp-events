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
    detection_latency_sec: float = 0.0  # peak_time − gt center
    gt_band: str | None = None  # band label of the ground truth event


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
    mean_detection_latency_sec: float
    median_period_error_pct: float
    f1_at_iou: dict[float, float] = field(default_factory=dict)
    matches: list[MatchResult] = field(default_factory=list)
    # Per-band GT counts and band-correct TP counts (unconditional
    # band accuracy, used for the macro-averaged suite metric).
    per_band_gt: dict[str, int] = field(default_factory=dict)
    per_band_correct: dict[str, int] = field(default_factory=dict)


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
    overall_f1_at_iou: dict[float, float] = field(default_factory=dict)
    dataset_scores: list[BenchmarkScore] = field(default_factory=list)
    # Macro-averaged band accuracy across {QP30, QP60, QP120}, where
    # each band's score is (band-correct TPs) / (injected detectable GT
    # events in that band). Unlike ``band_accuracy`` (which is TP-
    # conditional), this penalizes missed events against band quality.
    band_accuracy_macro: float = 1.0
    # Per-band recall (band-correct TPs / GT events in that band).
    per_band_accuracy: dict[str, float] = field(default_factory=dict)


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
    iou_thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
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
        Primary IoU threshold for match acceptance.
    iou_thresholds : tuple of float
        Additional IoU thresholds for multi-threshold F1 reporting
        (THUMOS14 convention).
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
        det_ranges = [_det_time_range(det, t0_sec) for det in detections]
        iou_matrix = np.zeros((n_gt, n_det))
        for i, gt in enumerate(gt_detectable):
            for j, (det_start, det_end) in enumerate(det_ranges):
                iou_matrix[i, j] = _time_iou(
                    gt.start_sec, gt.end_sec, det_start, det_end
                )

        # Hungarian assignment (maximise IoU → minimise negative IoU)
        cost = -iou_matrix
        gt_idx, det_idx = linear_sum_assignment(cost)

        # Build all matched pairs from Hungarian assignment
        all_pairs: list[tuple[int, int, float]] = []
        for gi, di in zip(gt_idx, det_idx):
            all_pairs.append((gi, di, float(iou_matrix[gi, di])))

        # Primary matches at the main threshold
        matches: list[MatchResult] = []
        matched_det: set[int] = set()
        matched_gt: set[int] = set()

        for gi, di, iou_val in all_pairs:
            if iou_val >= iou_threshold:
                gt_ev = gt_detectable[gi]
                det_ev = detections[di]
                det_period = det_ev.period_sec or 0.0
                period_err = (
                    abs(det_period - gt_ev.period_sec)
                    / gt_ev.period_sec * 100
                    if gt_ev.period_sec > 0
                    else 0.0
                )
                band_ok = (
                    det_ev.band == gt_ev.band if gt_ev.band else True
                )
                # Detection latency: peak_time − gt center
                det_peak_sec = (
                    det_ev.peak_time.timestamp() - t0_sec
                )
                latency = det_peak_sec - gt_ev.center_sec
                matches.append(MatchResult(
                    gt_event_id=gt_ev.event_id,
                    detected_idx=di,
                    iou=iou_val,
                    period_error_pct=period_err,
                    band_correct=band_ok,
                    detection_latency_sec=latency,
                    gt_band=gt_ev.band,
                ))
                matched_det.add(di)
                matched_gt.add(gi)

        # Multi-threshold F1 (reuse same Hungarian assignment)
        f1_at_iou: dict[float, float] = {}
        for thr in iou_thresholds:
            tp_t = sum(1 for _, _, v in all_pairs if v >= thr)
            fp_t = n_det - tp_t
            fn_t = n_gt - tp_t
            p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 1.0
            r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 1.0
            d_t = p_t + r_t
            f1_at_iou[thr] = 2 * p_t * r_t / d_t if d_t > 0 else 0.0
    else:
        matches = []
        matched_det = set()
        matched_gt = set()
        f1_at_iou = {thr: 0.0 for thr in iou_thresholds}

    tp = len(matches)
    fp = n_det - len(matched_det)
    fn = n_gt - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    band_acc = (
        sum(1 for m in matches if m.band_correct) / tp
        if tp > 0 else 1.0
    )
    mean_period_err = (
        float(np.mean([m.period_error_pct for m in matches]))
        if matches else 0.0
    )
    median_period_err = (
        float(np.median([m.period_error_pct for m in matches]))
        if matches else 0.0
    )
    mean_iou = (
        float(np.mean([m.iou for m in matches])) if matches else 0.0
    )
    mean_latency = (
        float(np.mean([m.detection_latency_sec for m in matches]))
        if matches else 0.0
    )

    # Per-difficulty recall
    recall_by_diff: dict[str, float] = {}
    for diff in ("easy", "moderate", "hard", "extreme"):
        gt_diff = [e for e in gt_detectable if e.difficulty == diff]
        if not gt_diff:
            continue
        matched_diff = sum(
            1 for gi in matched_gt
            if gt_detectable[gi].difficulty == diff
        )
        recall_by_diff[diff] = matched_diff / len(gt_diff)

    # Decoy rejection: count how many decoys were falsely detected.
    #
    # Two regimes:
    #   * Explicit decoys (gt_decoys non-empty): count IoU-matched FPs.
    #   * Pure-noise decoy scenarios (no injected events at all, e.g.
    #     ``decoy_red_noise_qp120``, ``decoy_ppo_harmonic``,
    #     ``decoy_roll_artifacts``): treat *any* detection as a
    #     rejection failure. Without this, these scenarios auto-score
    #     1.0 regardless of detector behaviour.
    n_decoy = len(gt_decoys)
    decoy_detected = 0
    if n_decoy > 0 and n_det > 0:
        if not (n_gt > 0):
            det_ranges = [
                _det_time_range(det, t0_sec) for det in detections
            ]
        for decoy in gt_decoys:
            for det_start, det_end in det_ranges:
                iou_d = _time_iou(
                    decoy.start_sec, decoy.end_sec,
                    det_start, det_end,
                )
                if iou_d >= iou_threshold:
                    decoy_detected += 1
                    break
        decoy_rejection = 1.0 - decoy_detected / n_decoy
    elif n_gt == 0 and manifest.difficulty_tier == "decoy":
        # Pure-noise decoy scenario: any detection is a false positive.
        # Convert FP count to a bounded rate: 0 FPs → 1.0, ≥ 4 FPs → 0.0.
        # Matches the order-of-magnitude of typical scenario event counts.
        decoy_detected = n_det
        decoy_rejection = max(0.0, 1.0 - n_det / 4.0)
        # n_decoy stays 0 (no injected decoys), but we report n_det via
        # n_decoy_detected so the scoreboard sees the penalty.
    else:
        decoy_rejection = 1.0

    # Per-band GT counts (for macro-averaged band accuracy downstream).
    per_band_gt: dict[str, int] = {}
    per_band_correct: dict[str, int] = {}
    for ev in gt_detectable:
        if ev.band is None:
            continue
        per_band_gt[ev.band] = per_band_gt.get(ev.band, 0) + 1
    for m in matches:
        if m.band_correct and m.gt_band is not None:
            per_band_correct[m.gt_band] = (
                per_band_correct.get(m.gt_band, 0) + 1
            )

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
        mean_detection_latency_sec=mean_latency,
        median_period_error_pct=median_period_err,
        per_band_gt=per_band_gt,
        per_band_correct=per_band_correct,
        f1_at_iou=f1_at_iou,
        matches=matches,
    )


def score_suite(scores: list[BenchmarkScore]) -> SuiteScore:
    """Aggregate scores across the entire benchmark suite.

    Diagnostic scenarios (prefix ``diag_``) are excluded from the
    aggregate metrics but kept in the returned ``dataset_scores`` so
    they can be inspected per-scenario.
    """
    all_scores = scores  # keep full list, including diag_*
    # Only aggregate non-diagnostic scenarios.
    scores = [s for s in all_scores if not s.dataset_id.startswith("diag_")]

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

    # Unconditional per-band accuracy:
    #   (band-correct TPs in band b) / (injected GT in band b)
    # Missed events count against the band — unlike the TP-conditional
    # ``band_accuracy`` below, which only sees detected events.
    per_band_gt_total: dict[str, int] = {}
    per_band_correct_total: dict[str, int] = {}
    for s in scores:
        for band, n in s.per_band_gt.items():
            per_band_gt_total[band] = per_band_gt_total.get(band, 0) + n
        for band, n in s.per_band_correct.items():
            per_band_correct_total[band] = (
                per_band_correct_total.get(band, 0) + n
            )
    per_band: dict[str, float] = {
        band: per_band_correct_total.get(band, 0) / n
        for band, n in per_band_gt_total.items()
        if n > 0
    }
    band_accuracy_macro = (
        float(np.mean(list(per_band.values()))) if per_band else 1.0
    )

    # TP-conditional band accuracy (kept for back-compat).
    total_matched = sum(len(s.matches) for s in scores)
    band_acc = (
        sum(sum(1 for m in s.matches if m.band_correct) for s in scores)
        / total_matched
        if total_matched > 0
        else 1.0
    )

    # Decoy rejection — macro-average across decoy datasets so that
    # pure-noise scenarios (n_decoy_events == 0 but decoy_rejection_rate
    # penalized for FPs) contribute on equal footing with explicit-decoy
    # scenarios. Falls back to 1.0 if no decoy datasets were scored.
    decoy_scores = [s for s in scores if s.dataset_id.startswith("decoy")]
    decoy_rej = (
        float(np.mean([s.decoy_rejection_rate for s in decoy_scores]))
        if decoy_scores
        else 1.0
    )

    # Summary: harmonic mean of F1, band_accuracy, decoy_rejection
    components = [x for x in (f1, band_acc, decoy_rej) if x > 0]
    summary = (
        len(components) / sum(1.0 / c for c in components)
        if components
        else 0.0
    )

    # Aggregate multi-threshold F1 across datasets
    all_thresholds: set[float] = set()
    for s in scores:
        all_thresholds.update(s.f1_at_iou.keys())
    overall_f1_at_iou: dict[float, float] = {}
    for thr in sorted(all_thresholds):
        # Macro-average of per-dataset F1 (exact micro-average would
        # need raw TP/FP/FN counts per threshold, not stored)
        vals = [s.f1_at_iou.get(thr, 0.0) for s in scores if s.f1_at_iou]
        overall_f1_at_iou[thr] = (
            float(np.mean(vals)) if vals else 0.0
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
        overall_f1_at_iou=overall_f1_at_iou,
        dataset_scores=all_scores,
        band_accuracy_macro=band_accuracy_macro,
        per_band_accuracy=per_band,
    )


def _non_diagnostic_scores(
    suite: SuiteScore,
) -> list[BenchmarkScore]:
    """Return dataset scores excluding diagnostic (``diag_*``) scenarios."""
    return [
        s for s in suite.dataset_scores
        if not s.dataset_id.startswith("diag_")
    ]


def composite_detection_score(suite: SuiteScore) -> float:
    r"""Weighted harmonic mean of detection capabilities.

    Diagnostic scenarios (``diag_*``) are excluded — they only
    contribute to the per-scenario breakdown. Weights reflect
    scientific priorities: detection (F1) first, then frequency
    accuracy (band + period), then localization and specificity.

    $$\text{score} = \frac{\sum w_i}
    {\sum w_i / \max(c_i, \epsilon)}$$
    """
    dataset_scores = _non_diagnostic_scores(suite)
    # Clamp period error to [0, 100] then convert to accuracy
    total_matched = sum(len(s.matches) for s in dataset_scores)
    if total_matched > 0:
        mean_period_err = float(np.mean([
            m.period_error_pct
            for s in dataset_scores
            for m in s.matches
        ]))
    else:
        mean_period_err = 100.0
    period_acc = max(0.0, 1.0 - mean_period_err / 100.0)

    mean_iou = float(np.mean([
        s.mean_iou for s in dataset_scores if s.mean_iou > 0
    ])) if any(s.mean_iou > 0 for s in dataset_scores) else 0.0

    # Band component is the macro-averaged, unconditional accuracy —
    # missed events count against their band, preventing a detector
    # from inflating the score by dropping low-confidence detections.
    components = [
        (0.35, suite.overall_f1),
        (0.20, suite.band_accuracy_macro),
        (0.15, period_acc),
        (0.15, suite.decoy_rejection_rate),
        (0.15, mean_iou),
    ]

    eps = 1e-10
    w_sum = sum(w for w, _ in components)
    denom = sum(w / max(c, eps) for w, c in components)
    return w_sum / denom if denom > 0 else 0.0
