"""Hybrid classifier: regulatorily-defensible rules engine + learned override.

Two layers:

  Rules engine — hard thresholds on total specific activity (Bq/g) and dose
                 rate (uSv/h at 30 cm). Auditable, regulator-friendly.
                 Documented in data/iaea_thresholds.json.

  ML head      — a tiny k-NN on photopeak-window features. Used for
                 *confidence estimation* and *spike detection*. Specifically,
                 the coordinator derives a learned threshold on the
                 "actinide-window normalized counts" feature from HPGe
                 rescrutiny labels. When applied, this catches items that
                 fall inside the LLW activity envelope but contain an
                 actinide spike that should put them in ILW (the tricky case).

The discriminating learned quantity is intentionally simple — a single scalar
threshold — so the live demo can clearly show "before learning: X / after
learning: Y" and so the coordinator can compute it from a small handful of
trusted rescrutiny labels.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .sensors import CHANNEL_CENTERS


# Photopeak windows used as features (keV low, keV high)
PHOTOPEAK_WINDOWS: dict[str, tuple[float, float]] = {
    "Am-241_60":  (50,   75),
    "U-235_186":  (170,  205),
    "I-131_365":  (340,  390),
    "Pu-239_414": (395,  430),
    "Cs-134_605": (585,  625),
    "Cs-137_662": (640,  685),
    "Cs-134_796": (775,  815),
    "Mn-54_835":  (820,  850),
    "Co-58_811":  (800,  825),
    "U-238_1001": (985, 1020),
    "Co-60_1173": (1150, 1195),
    "Co-60_1332": (1310, 1355),
}

ACTINIDE_WINDOW_NAMES = ("Am-241_60", "U-235_186", "Pu-239_414", "U-238_1001")


@dataclass
class ClassificationResult:
    predicted_class: str            # VLLW / LLW / ILW
    confidence: float
    rule_class: str
    override_applied: bool          # True if learned threshold escalated to ILW
    actinide_signature: float
    spectrum_features: np.ndarray
    cv_class: str | None = None
    cv_agreed: bool = True


THRESHOLDS_PATH = Path(__file__).parent.parent / "data" / "iaea_thresholds.json"
with open(THRESHOLDS_PATH) as f:
    _THR = json.load(f)["class_boundaries"]

VLLW_SA_MAX = _THR["VLLW_max_specific_activity_bq_per_g"]
LLW_SA_MAX = _THR["LLW_max_specific_activity_bq_per_g"]
VLLW_DR_MAX = _THR["VLLW_max_dose_rate_uSv_h"]
LLW_DR_MAX = _THR["LLW_max_dose_rate_uSv_h"]


def extract_spectrum_features(spectrum: np.ndarray) -> np.ndarray:
    """Normalize photopeak-window counts to total spectrum counts so the
    feature is independent of integration time / source strength."""
    total = max(spectrum.sum(), 1.0)
    features = []
    for lo, hi in PHOTOPEAK_WINDOWS.values():
        mask = (CHANNEL_CENTERS >= lo) & (CHANNEL_CENTERS < hi)
        features.append(spectrum[mask].sum() / total)
    return np.asarray(features, dtype=np.float64)


def actinide_signature(features: np.ndarray) -> float:
    names = list(PHOTOPEAK_WINDOWS.keys())
    idx = [i for i, n in enumerate(names) if n in ACTINIDE_WINDOW_NAMES]
    return float(features[idx].sum())


def rules_classify(specific_activity_bq_per_g: float, dose_rate_uSv_h: float) -> str:
    if specific_activity_bq_per_g <= VLLW_SA_MAX and dose_rate_uSv_h <= VLLW_DR_MAX:
        return "VLLW"
    if specific_activity_bq_per_g <= LLW_SA_MAX and dose_rate_uSv_h <= LLW_DR_MAX:
        return "LLW"
    return "ILW"


def _rule_confidence(specific_activity_bq_per_g: float, rule_class: str) -> float:
    """Confidence is higher when the specific activity sits well inside a
    class range, lower near the class boundary. Floored at 0.72 (just
    above the 0.70 scrutiny threshold) so a typical item is NOT flagged
    on rules alone — only items right at the boundary, or those where CV
    or the ML head disagrees, drop below threshold and go to the QA lab."""
    if specific_activity_bq_per_g <= 0:
        return 0.72
    log_sa = np.log10(specific_activity_bq_per_g)
    log_vllw = np.log10(VLLW_SA_MAX)
    log_llw = np.log10(LLW_SA_MAX)
    if rule_class == "VLLW":
        margin = log_vllw - log_sa
    elif rule_class == "LLW":
        margin = min(log_sa - log_vllw, log_llw - log_sa)
    else:
        margin = log_sa - log_llw
    # 0 margin -> 0.72; >2 decades from boundary -> ~0.92
    return float(np.clip(0.72 + 0.10 * margin, 0.68, 0.95))


class HybridClassifier:
    """Hybrid classifier. Threshold-based actinide-spike override is the
    primary learned quantity — coordinator computes it from HPGe rescrutiny
    labels and pushes it to all agents in hivemind mode. ML head supplements
    confidence; in isolated mode neither is updated."""

    def __init__(self):
        self._scaler: StandardScaler | None = None
        self._knn: KNeighborsClassifier | None = None
        self._trained = False
        # The learned actinide-signature threshold. None means "no override";
        # otherwise: if rule says LLW/VLLW but actinide_sig >= threshold,
        # escalate to ILW.
        self.actinide_threshold: float | None = None
        self.training_samples_seen = 0

    # ---------- Training (coordinator side) ----------

    def fit(
        self,
        features: np.ndarray,
        labels: list[str],
        actinide_threshold: float | None,
    ) -> None:
        """Re-fit ML head and accept a (separately computed) actinide
        threshold. Designed for <1s on ~10^3 samples."""
        self.actinide_threshold = actinide_threshold
        if len(set(labels)) < 2 or len(labels) < 5:
            self._trained = False
            self.training_samples_seen = len(labels)
            return
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(features)
        self._knn = KNeighborsClassifier(n_neighbors=min(5, max(1, len(labels) // 3)))
        self._knn.fit(Xs, labels)
        self._trained = True
        self.training_samples_seen = len(labels)

    # ---------- Inference ----------

    def classify(
        self,
        specific_activity_bq_per_g: float,
        dose_rate_uSv_h: float,
        spectrum: np.ndarray,
        cv_probs: dict[str, float] | None = None,
    ) -> ClassificationResult:
        features = extract_spectrum_features(spectrum)
        rule_class = rules_classify(specific_activity_bq_per_g, dose_rate_uSv_h)
        sig = actinide_signature(features)

        override_applied = False
        if (
            rule_class != "ILW"
            and self.actinide_threshold is not None
            and sig >= self.actinide_threshold
        ):
            predicted = "ILW"
            override_applied = True
            confidence = 0.85
        else:
            predicted = rule_class
            confidence = _rule_confidence(specific_activity_bq_per_g, rule_class)

        # ML head adjusts confidence (if trained and disagrees, knock it
        # down by a moderate additive amount). Strong rule-confidence items
        # survive; borderline ones drop below the scrutiny threshold.
        if self._trained and self._knn is not None and self._scaler is not None:
            Xs = self._scaler.transform(features.reshape(1, -1))
            proba = self._knn.predict_proba(Xs)[0]
            classes = list(self._knn.classes_)
            ml_class = classes[int(np.argmax(proba))]
            if ml_class != predicted:
                confidence = max(confidence - 0.12, 0.40)

        # Items right at the actinide threshold get scrutinized — this is
        # how the hivemind keeps generating training data near the decision
        # boundary. Window kept very tight so this isn't an everyday trigger.
        if self.actinide_threshold is not None and predicted != "ILW":
            margin = self.actinide_threshold - sig
            if 0 <= margin < 0.005:
                confidence = min(confidence, 0.65)

        # CV cross-check. If the camera-based class disagrees with the
        # gamma+rules prediction, knock confidence down by a fixed amount.
        # Strong gamma calls (conf ~0.95) survive the penalty; borderline
        # ones (conf ~0.75) drop below the scrutiny threshold and get sent
        # to the QA lab. CV cannot catch the actinide-spike trick on its
        # own (the trick *is* an LLW-looking container).
        cv_class = None
        cv_agreed = True
        if cv_probs is not None:
            cv_class = max(cv_probs, key=cv_probs.get)
            cv_agreed = (cv_class == predicted)
            if not cv_agreed:
                confidence = max(confidence - 0.18, 0.40)

        return ClassificationResult(
            predicted_class=predicted,
            confidence=confidence,
            rule_class=rule_class,
            override_applied=override_applied,
            actinide_signature=sig,
            spectrum_features=features,
            cv_class=cv_class,
            cv_agreed=cv_agreed,
        )

    # ---------- Snapshot / restore ----------

    def state_snapshot(self) -> dict:
        snap: dict = {
            "actinide_threshold": self.actinide_threshold,
            "training_samples_seen": self.training_samples_seen,
            "trained": self._trained,
        }
        if self._trained and self._knn is not None and self._scaler is not None:
            original_labels = [self._knn.classes_[i] for i in self._knn._y.tolist()]
            snap.update({
                "scaler_mean": self._scaler.mean_.tolist(),
                "scaler_scale": self._scaler.scale_.tolist(),
                "knn_X": self._knn._fit_X.tolist(),
                "knn_y_labels": original_labels,
                "n_neighbors": self._knn.n_neighbors,
            })
        return snap

    def load_snapshot(self, snap: dict) -> None:
        self.actinide_threshold = snap.get("actinide_threshold")
        self.training_samples_seen = snap.get("training_samples_seen", 0)
        if not snap.get("trained"):
            self._trained = False
            return
        self._scaler = StandardScaler()
        self._scaler.mean_ = np.asarray(snap["scaler_mean"])
        self._scaler.scale_ = np.asarray(snap["scaler_scale"])
        self._scaler.var_ = self._scaler.scale_ ** 2
        self._scaler.n_features_in_ = len(snap["scaler_mean"])
        knn = KNeighborsClassifier(n_neighbors=snap["n_neighbors"])
        X = np.asarray(snap["knn_X"])
        y = list(snap["knn_y_labels"])
        knn.fit(X, y)
        self._knn = knn
        self._trained = True


def derive_actinide_threshold(
    features: np.ndarray,
    labels: list[str],
    min_samples_per_class: int = 3,
    fp_budget: float = 0.02,
) -> float | None:
    """Compute the discriminating actinide-signature threshold from a labeled
    training corpus.

    The decision metric is asymmetric: a false negative (under-classifying
    ILW) is a regulatory/safety problem; a false positive (over-classifying
    LLW to ILW) is "only" expensive storage. So we choose the threshold that
    maximizes true-positive rate subject to false-positive rate <= fp_budget.

    If no threshold satisfies the budget (the distributions overlap heavily),
    return p99(non-ILW) + a small margin — the most permissive
    actinide-spike detector we can ship safely."""
    if len(labels) == 0:
        return None
    sigs = np.array([actinide_signature(f) for f in features])
    labels_arr = np.array(labels)
    ilw_sigs = sigs[labels_arr == "ILW"]
    non_ilw_sigs = sigs[(labels_arr == "LLW") | (labels_arr == "VLLW")]
    if len(ilw_sigs) < min_samples_per_class or len(non_ilw_sigs) < min_samples_per_class:
        return None
    # Candidate thresholds: midpoints between sorted unique signature values
    candidates = np.unique(np.concatenate([ilw_sigs, non_ilw_sigs]))
    best_thr = None
    best_tpr = -1.0
    for thr in candidates:
        tpr = float(np.mean(ilw_sigs >= thr))
        fpr = float(np.mean(non_ilw_sigs >= thr))
        if fpr <= fp_budget and tpr > best_tpr:
            best_tpr = tpr
            best_thr = float(thr)
    if best_thr is None:
        return float(np.percentile(non_ilw_sigs, 99)) + 0.005
    return best_thr
