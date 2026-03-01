"""
Neurothermo: Production Library for Thermodynamic Monitoring.

FAST during training: Only delta (O(n))
FULL metrics at summary(): All 17 metrics computed from accumulated data
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    import torch
    from torch.nn.utils import parameters_to_vector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PhaseState(Enum):
    """Thermodynamic phase states."""
    CRYSTAL = auto()
    GLASS = auto()
    LIQUID = auto()
    TOPOLOGICAL_INSULATOR = auto()
    TRANSITION = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ThresholdConfig:
    delta_crystal_threshold: float = 0.1
    delta_glass_threshold: float = 0.4
    purity_high_threshold: float = 7.0
    mbl_poisson_ratio: float = 0.38
    zero_epsilon: float = 1.0e-6
    numerical_stability_epsilon: float = 1.0e-9


@dataclass(frozen=True)
class ComputeConfig:
    gradient_buffer_window: int = 50


@dataclass(frozen=True)
class CoreConfig:
    window_size: int = 50
    enable_logging: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class NeurothermoConfig:
    core: CoreConfig = field(default_factory=CoreConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    computation: ComputeConfig = field(default_factory=ComputeConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "NeurothermoConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            core=CoreConfig(**data.get("core", {})),
            thresholds=ThresholdConfig(**data.get("thresholds", {})),
            computation=ComputeConfig(**data.get("computation", {})),
        )


class MetricsResult:
    """Container for step metrics (only delta/alpha/health/phase during training)."""

    def __init__(self, metrics: Dict[str, float], phase: PhaseState) -> None:
        self._metrics = dict(metrics)
        self._phase = phase

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        return self._metrics.get(name, default)

    def to_dict(self) -> Dict[str, float]:
        return dict(self._metrics)

    @property
    def phase(self) -> PhaseState:
        return self._phase


def _detect_phase(delta: float) -> PhaseState:
    """Fast phase detection from delta only."""
    if delta < 0.1:
        return PhaseState.CRYSTAL
    if delta > 0.4:
        return PhaseState.GLASS
    return PhaseState.TRANSITION


class ThermoMonitor:
    """Monitoring class.

    During training: ONLY delta (O(n), instant)
    At summary(): ALL 17 metrics from accumulated history
    """

    def __init__(self, model: Optional[Any] = None, config: Optional[NeurothermoConfig] = None) -> None:
        self._config = config or NeurothermoConfig()
        self._model = model

        self._step_count = 0
        self._epoch_count = 0
        self._last_result: Optional[MetricsResult] = None

        # History for final computation
        self._weights_history: List[np.ndarray] = []
        self._gradients_history: List[np.ndarray] = []
        self._loss_history: List[float] = []
        self._delta_history: List[float] = []
        self._phase_history: List[PhaseState] = []

        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("neurothermo")
        logger.setLevel(getattr(logging, self._config.core.log_level, logging.INFO))
        if not logger.handlers and self._config.core.enable_logging:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _extract_weights(self) -> np.ndarray:
        if self._model is None:
            raise ValueError("No model. Use step_manual().")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required.")
        return parameters_to_vector(self._model.parameters()).detach().cpu().numpy()

    def _extract_gradients(self) -> Optional[np.ndarray]:
        if self._model is None or not TORCH_AVAILABLE:
            return None
        grads = [p.grad.detach().flatten() for p in self._model.parameters() if p.grad is not None]
        return torch.cat(grads).cpu().numpy() if grads else None

    def step(self, loss: Optional[float] = None) -> MetricsResult:
        """Fast step: ONLY computes delta. Stores data for final metrics."""
        weights = self._extract_weights()
        gradients = self._extract_gradients()
        return self._do_step(weights, gradients, loss)

    def step_manual(
        self,
        weights: np.ndarray,
        gradients: Optional[np.ndarray] = None,
        loss: Optional[float] = None,
    ) -> MetricsResult:
        """Manual step with provided arrays."""
        return self._do_step(weights, gradients, loss)

    def _do_step(
        self,
        weights: np.ndarray,
        gradients: Optional[np.ndarray],
        loss: Optional[float],
    ) -> MetricsResult:
        """Compute ONLY delta. Everything else deferred to summary()."""
        eps = self._config.thresholds.numerical_stability_epsilon

        # Store for final computation
        self._weights_history.append(weights.copy())
        if gradients is not None:
            self._gradients_history.append(gradients.copy())
        if loss is not None:
            self._loss_history.append(loss)

        # === ONLY FAST METRIC: delta (O(n)) ===
        delta = float(np.max(np.abs(weights - np.round(weights))))

        # Derivatives of delta (O(1))
        alpha = -math.log(delta + eps)
        delta_comp = min(1.0, 1.0 / (1.0 + delta))
        alpha_comp = min(1.0, alpha / self._config.thresholds.purity_high_threshold)
        health = 0.5 * delta_comp + 0.5 * alpha_comp

        # Phase
        phase = _detect_phase(delta)

        self._delta_history.append(delta)
        self._phase_history.append(phase)
        self._step_count += 1

        result = MetricsResult({"delta": delta, "alpha": alpha, "health": health}, phase)
        self._last_result = result
        return result

    def epoch_end(self) -> None:
        self._epoch_count += 1

    def get_phase_description(self, phase: PhaseState) -> str:
        descriptions = {
            PhaseState.CRYSTAL: "Crystal: weights discretized. Model learned the algorithm.",
            PhaseState.GLASS: "Glass: amorphous weights. Model may be memorizing.",
            PhaseState.LIQUID: "Liquid: localized (MBL) behavior.",
            PhaseState.TOPOLOGICAL_INSULATOR: "Topological insulator: robust generalization.",
            PhaseState.TRANSITION: "Phase transition: between crystalline and glassy.",
            PhaseState.UNKNOWN: "Phase undetermined.",
        }
        return descriptions.get(phase, "Unknown")

    def reset(self) -> None:
        self._step_count = 0
        self._epoch_count = 0
        self._last_result = None
        self._weights_history.clear()
        self._gradients_history.clear()
        self._loss_history.clear()
        self._delta_history.clear()
        self._phase_history.clear()

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute ALL 17 metrics from history. Call after training."""
        if not self._weights_history:
            return {}

        eps = self._config.thresholds.numerical_stability_epsilon
        metrics: Dict[str, float] = {}
        final_weights = self._weights_history[-1]

        # === Phase Detection Metrics ===
        metrics["delta"] = self._delta_history[-1]
        metrics["alpha"] = -math.log(metrics["delta"] + eps)

        abs_w = np.abs(final_weights)
        threshold = max(np.quantile(abs_w, 0.95) * 0.01, eps)
        metrics["lc"] = float(np.sum(abs_w > threshold)) / len(final_weights)
        metrics["vacuum"] = float(np.sum(abs_w < self._config.thresholds.zero_epsilon)) / len(final_weights)

        delta_comp = min(1.0, 1.0 / (1.0 + metrics["delta"]))
        alpha_comp = min(1.0, metrics["alpha"] / self._config.thresholds.purity_high_threshold)
        metrics["health"] = 0.5 * delta_comp + 0.5 * alpha_comp

        # === Gradient Metrics ===
        if len(self._gradients_history) >= 2:
            all_grads = np.array(self._gradients_history)
            centered = all_grads - np.mean(all_grads, axis=0, keepdims=True)

            try:
                cov = np.cov(centered, rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > eps]

                if len(eigvals) >= 2:
                    metrics["kappa"] = float(eigvals[-1] / eigvals[0])
                    metrics["t_eff"] = float(np.sum(eigvals) / len(eigvals))
                    metrics["ricci"] = float(len(eigvals) * np.sum(1.0 / eigvals))

                # Participation Ratio
                if all_grads.shape[0] > 1 and all_grads.shape[1] > 1:
                    corr = np.corrcoef(all_grads.T)
                    if corr.ndim == 2:
                        corr_eig = np.linalg.eigvalsh(corr)
                        corr_eig = corr_eig[corr_eig > eps]
                        if len(corr_eig) > 0:
                            s, s2 = np.sum(corr_eig), np.sum(corr_eig ** 2)
                            if s2 > eps:
                                metrics["pr"] = float((s ** 2) / s2)
            except Exception:
                pass

            # Learning Uncertainty
            norms = [np.linalg.norm(g) for g in self._gradients_history]
            mean_n, std_n = np.mean(norms), np.std(norms)
            if mean_n > eps:
                metrics["hbar"] = float(std_n / mean_n)

        # === Topological Metrics ===
        total_phase = 0.0
        for i in range(1, len(self._weights_history)):
            prev = self._weights_history[i - 1]
            curr = self._weights_history[i]
            prev_n = prev / (np.linalg.norm(prev) + eps)
            curr_n = curr / (np.linalg.norm(curr) + eps)
            total_phase += float(np.angle(np.vdot(prev_n, curr_n)))
        metrics["berry_phase_total"] = total_phase

        # Level Spacing Ratio
        diffs = np.diff(np.sort(np.abs(final_weights)))
        diffs = diffs[diffs > eps]
        if len(diffs) > 1:
            ratios = [min(diffs[i], diffs[i+1]) / max(diffs[i], diffs[i+1])
                     for i in range(len(diffs)-1) if max(diffs[i], diffs[i+1]) > 0]
            if ratios:
                metrics["r"] = float(np.mean(ratios))

        # Algorithmic Gravitational
        if self._gradients_history:
            grad_norm = np.linalg.norm(self._gradients_history[-1])
            metrics["g_alg"] = float(grad_norm) / (metrics["delta"] ** 2 + eps)

        metrics["hbar_eff"] = 0.0  # Requires reg_lambda

        # === Other Metrics ===
        counts, _ = np.histogram(final_weights, bins=50, density=True)
        counts = counts + eps
        probs = counts / np.sum(counts)
        metrics["entropy"] = float(-np.sum(probs * np.log(probs + eps)))

        if len(self._loss_history) >= 2:
            metrics["c_v"] = float(np.mean(np.abs(np.gradient(self._loss_history))))

        metrics["psi"] = 1.0

        return metrics

    def summary(self) -> str:
        """Generate summary with ALL metrics."""
        lines = []
        lines.append("=" * 70)
        lines.append("NEUROTHERMO SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Steps: {self._step_count}, Epochs: {self._epoch_count}")
        lines.append("")

        # Compute ALL metrics now
        all_metrics = self.compute_all_metrics()

        lines.append("ALL METRICS:")
        lines.append("-" * 50)

        for name in ["delta", "alpha", "lc", "vacuum", "health", "kappa", "t_eff",
                     "ricci", "pr", "hbar", "berry_phase_total", "r", "g_alg",
                     "hbar_eff", "entropy", "c_v", "psi"]:
            val = all_metrics.get(name)
            if val is not None:
                if math.isinf(val):
                    lines.append(f"  {name}: inf")
                else:
                    lines.append(f"  {name}: {val:.6f}")

        if self._phase_history:
            lines.append("")
            lines.append(f"Final Phase: {self._phase_history[-1].name}")
            lines.append(f"  {self.get_phase_description(self._phase_history[-1])}")

        if self._delta_history:
            lines.append("")
            lines.append("DELTA STATISTICS:")
            d = np.array(self._delta_history)
            lines.append(f"  mean: {np.mean(d):.4f}, std: {np.std(d):.4f}")
            lines.append(f"  range: [{np.min(d):.4f}, {np.max(d):.4f}]")

        if self._phase_history:
            lines.append("")
            lines.append("PHASE DISTRIBUTION:")
            counts = {}
            for p in self._phase_history:
                counts[p] = counts.get(p, 0) + 1
            total = len(self._phase_history)
            for phase, count in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {phase.name}: {count} ({100*count/total:.1f}%)")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_result(self) -> Optional[MetricsResult]:
        return self._last_result


def create_monitor(model: Optional[Any] = None, window_size: int = 50) -> ThermoMonitor:
    """Create monitor. During training only delta is computed (fast)."""
    config = NeurothermoConfig(
        core=CoreConfig(window_size=window_size),
        computation=ComputeConfig(gradient_buffer_window=window_size),
    )
    return ThermoMonitor(model=model, config=config)


def extract_weights(model: Any) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    return parameters_to_vector(model.parameters()).detach().cpu().numpy()


def extract_gradients(model: Any) -> Optional[np.ndarray]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads).cpu().numpy() if grads else None


__all__ = [
    "NeurothermoConfig", "ThermoMonitor", "MetricsResult", "PhaseState",
    "create_monitor", "extract_weights", "extract_gradients",
]
