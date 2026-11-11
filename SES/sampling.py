"""Equal-interval retention of whole 30 s epochs, plus the SES 1 s grid.

Deployment (``model.eval``) uses phase 0 at both scales. The kept epoch
indices and the kept 1 s indices are therefore a function of the budget
alone. Every 32-epoch window collects the same number of seconds, so the
battery cost of a night with T epochs can be written down before recording.

Training still follows the SES intra-epoch rule: shift the 1 s grid by an
integer phase inside one step, and with probability ``mix_random`` replace
that grid by a random intra-epoch subset. The epoch grid uses the same
phase rule. It is never a fresh random subset.

``random_inter`` keeps that intra-epoch rule and draws the epoch subset
uniformly.

Table-3 ablations keep the phase-0 epoch timetable and change only the 1 s
patches: ``random`` redraws them, ``head`` keeps the first seconds, and
``uniform`` keeps the phase-0 grid with no phase shift and no random mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .config import (
    NUM_EPOCHS_WINDOW,
    NUM_PATCHES,
    budget_to_mass_mask_ratios,
    canonical_budget,
    phase_span,
    visible_count,
)


@dataclass
class PolicyOutput:
    epoch_visible: torch.Tensor  # [B, Ev]
    epoch_masked: torch.Tensor  # [B, Em]
    patch_visible: torch.Tensor  # [B, Ev, Pv]
    freq_mask: Optional[torch.Tensor]
    time_ratio: float
    freq_ratio: float

    @property
    def overall_ratio(self) -> float:
        return self.time_ratio * self.freq_ratio


def _randperm_batch(num_tokens: int, batch_size: int, generator=None, device=None) -> torch.Tensor:
    rows = []
    for _ in range(batch_size):
        if generator is not None:
            rows.append(torch.randperm(num_tokens, generator=generator))
        else:
            rows.append(torch.randperm(num_tokens))
    out = torch.stack(rows, dim=0)
    return out.to(device) if device is not None else out


def _uniquify_row(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Make a 1-D index tensor unique while preserving order."""
    k = int(idx.numel())
    seen = set()
    out: List[int] = []
    for value in idx.tolist():
        iv = int(value) % n
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    fill = 0
    while len(out) < k:
        if fill not in seen:
            seen.add(fill)
            out.append(fill)
        fill += 1
        if fill >= n:
            break
    return torch.tensor(out[:k], device=idx.device, dtype=torch.long)


def strided_indices(keep: int, num_tokens: int, offset: int, device=None) -> torch.Tensor:
    """Equal-interval indices.

    ``idx[i] = floor((offset + i * N / K) mod N)``. Adjacent kept tokens
    differ by ``floor(N/K)`` or ``ceil(N/K)``.
    """
    if keep >= num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)
    keep = max(1, int(keep))
    step = num_tokens / float(keep)
    pos = (float(offset % num_tokens) + torch.arange(keep, device=device, dtype=torch.float32) * step) % num_tokens
    idx = pos.floor().long() % num_tokens
    return _uniquify_row(idx, num_tokens)


def masked_from_visible(visible: torch.Tensor, num_tokens: int) -> torch.Tensor:
    keep = torch.zeros(num_tokens, dtype=torch.bool, device=visible.device)
    keep[visible] = True
    return torch.arange(num_tokens, device=visible.device)[~keep]


def _integer_offset(keep: int, num_tokens: int, training: bool, device) -> int:
    span = phase_span(keep, num_tokens)
    if training and span > 1:
        return int(torch.randint(0, span, (1,), device=device).item())
    return 0


class ObservationPolicy(nn.Module):
    name = "base"

    def __init__(
        self,
        budget: float,
        num_epochs: int = NUM_EPOCHS_WINDOW,
        num_patches: int = NUM_PATCHES,
        seed: Optional[int] = 2025,
        mix_random: float = 0.0,
    ):
        super().__init__()
        self.budget = canonical_budget(budget)
        self.num_epochs = int(num_epochs)
        self.num_patches = int(num_patches)
        self.mix_random = float(mix_random)
        self.intra_mask, self.inter_mask = budget_to_mass_mask_ratios(self.budget)
        self.keep_patches = visible_count(self.intra_mask, self.num_patches)
        self.keep_epochs = visible_count(self.inter_mask, self.num_epochs)
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))
        self.last_output: Optional[PolicyOutput] = None

    def apply_frequency(self, eeg: torch.Tensor, freq_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if freq_mask is None:
            return eeg
        mask = freq_mask.to(device=eeg.device, dtype=eeg.dtype).view(1, 1, 1, -1)
        return eeg * mask

    def sample(self, batch_size: int, device, eeg: Optional[torch.Tensor] = None) -> PolicyOutput:
        raise NotImplementedError

    def _random_patches(self, batch_size: int, n_vis_epochs: int, device) -> torch.Tensor:
        k = self.keep_patches
        perm = _randperm_batch(self.num_patches, batch_size * n_vis_epochs, self.generator, device)
        return perm[:, :k].view(batch_size, n_vis_epochs, k)

    def _maybe_mix_random(self, structured: torch.Tensor, batch_size: int, n_vis_epochs: int, device) -> torch.Tensor:
        if (not self.training) or self.mix_random <= 0:
            return structured
        if torch.rand((), device=device).item() < self.mix_random:
            return self._random_patches(batch_size, n_vis_epochs, device)
        return structured

    def _random_epochs(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_len = self.num_epochs - self.keep_epochs
        if mask_len <= 0:
            visible = torch.arange(self.num_epochs, device=device).unsqueeze(0).expand(batch_size, -1)
            masked = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return visible.contiguous(), masked
        perm = _randperm_batch(self.num_epochs, batch_size, self.generator, device)
        masked = perm[:, :mask_len]
        visible = perm[:, mask_len:]
        return visible, masked

    def _equal_interval_epochs(self, batch_size: int, device, allow_train_phase: bool):
        """Shared whole-epoch grid. Ablations pass ``allow_train_phase=False``."""
        offset = _integer_offset(
            self.keep_epochs, self.num_epochs, bool(self.training and allow_train_phase), device,
        )
        epoch_idx = strided_indices(self.keep_epochs, self.num_epochs, offset, device)
        epoch_idx = torch.sort(epoch_idx).values
        if int(epoch_idx.numel()) != self.keep_epochs or int(torch.unique(epoch_idx).numel()) != self.keep_epochs:
            raise RuntimeError(
                f"Epoch grid is not a unique keep-{self.keep_epochs} subset (offset={offset})."
            )
        epoch_msk_idx = masked_from_visible(epoch_idx, self.num_epochs)
        epoch_vis = epoch_idx.view(1, -1).expand(batch_size, -1).contiguous()
        epoch_msk = epoch_msk_idx.view(1, -1).expand(batch_size, -1).contiguous()
        return epoch_vis, epoch_msk

    def _structured_patches(self, batch_size: int, n_vis_epochs: int, offset: int, device) -> torch.Tensor:
        patch_idx = strided_indices(self.keep_patches, self.num_patches, offset, device)
        if int(patch_idx.numel()) != self.keep_patches or int(torch.unique(patch_idx).numel()) != self.keep_patches:
            raise RuntimeError(
                f"Patch grid is not a unique keep-{self.keep_patches} subset (offset={offset})."
            )
        return patch_idx.view(1, 1, -1).expand(batch_size, n_vis_epochs, -1).contiguous()

    def _pack(self, epoch_vis, epoch_msk, patch_vis) -> PolicyOutput:
        time_ratio = (epoch_vis.shape[1] / self.num_epochs) * (patch_vis.shape[-1] / self.num_patches)
        out = PolicyOutput(epoch_vis, epoch_msk, patch_vis, None, time_ratio, 1.0)
        self.last_output = out
        return out


class EqualIntervalPolicy(ObservationPolicy):
    """Equal-interval epochs and equal-interval 1 s patches. Default SES policy."""

    name = "equal"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._equal_interval_epochs(batch_size, device, allow_train_phase=True)
        patch_offset = _integer_offset(self.keep_patches, self.num_patches, self.training, device)
        patch_vis = self._structured_patches(batch_size, self.keep_epochs, patch_offset, device)
        patch_vis = self._maybe_mix_random(patch_vis, batch_size, self.keep_epochs, device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class RandomIntraPolicy(ObservationPolicy):
    """SES-Random: phase-0 equal-interval epochs, random 1 s patches at train and eval."""

    name = "random"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._equal_interval_epochs(batch_size, device, allow_train_phase=False)
        patch_vis = self._random_patches(batch_size, epoch_vis.shape[1], device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class HeadPolicy(ObservationPolicy):
    """SES-Head: phase-0 equal-interval epochs, first kept seconds of each epoch."""

    name = "head"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._equal_interval_epochs(batch_size, device, allow_train_phase=False)
        patch_idx = torch.arange(self.keep_patches, device=device)
        patch_vis = patch_idx.view(1, 1, -1).expand(batch_size, epoch_vis.shape[1], -1).contiguous()
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class UniformPolicy(ObservationPolicy):
    """SES-Uniform: phase-0 grids at both scales, with no phase shift and no random mix."""

    name = "uniform"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._equal_interval_epochs(batch_size, device, allow_train_phase=False)
        patch_vis = self._structured_patches(batch_size, epoch_vis.shape[1], 0, device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class RandomInterPolicy(ObservationPolicy):
    """SES 1 s grid with a random whole-epoch mask."""

    name = "random_inter"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._random_epochs(batch_size, device)
        patch_offset = _integer_offset(self.keep_patches, self.num_patches, self.training, device)
        patch_idx = strided_indices(self.keep_patches, self.num_patches, patch_offset, device)
        patch_vis = patch_idx.view(1, 1, -1).expand(batch_size, epoch_vis.shape[1], -1).contiguous()
        patch_vis = self._maybe_mix_random(patch_vis, batch_size, epoch_vis.shape[1], device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


def build_policy(
    name: str,
    budget: float,
    mix_random: float = 0.25,
    **kwargs,
) -> ObservationPolicy:
    name = name.lower()
    if name == "equal":
        return EqualIntervalPolicy(budget, mix_random=mix_random, **kwargs)
    if name == "random":
        return RandomIntraPolicy(budget, mix_random=0.0, **kwargs)
    if name == "head":
        return HeadPolicy(budget, mix_random=0.0, **kwargs)
    if name == "uniform":
        return UniformPolicy(budget, mix_random=0.0, **kwargs)
    if name == "random_inter":
        return RandomInterPolicy(budget, mix_random=mix_random, **kwargs)
    raise ValueError(
        f"Unknown policy {name}. Choose from equal, random, head, uniform, random_inter."
    )


def deployment_schedule(
    budget: float,
    num_epochs: int = NUM_EPOCHS_WINDOW,
    num_patches: int = NUM_PATCHES,
) -> Dict:
    """Phase-0 timetable. This is what ``model.eval`` acquires."""
    budget = canonical_budget(budget)
    policy = EqualIntervalPolicy(
        budget, num_epochs=num_epochs, num_patches=num_patches, mix_random=0.0, seed=None,
    )
    policy.eval()
    out = policy.sample(1, torch.device("cpu"))
    epochs = [int(v) for v in out.epoch_visible[0].tolist()]
    masked = [int(v) for v in out.epoch_masked[0].tolist()]
    patches = [int(v) for v in out.patch_visible[0, 0].tolist()]
    seconds = len(epochs) * len(patches)
    window_seconds = num_epochs * num_patches
    return {
        "budget_nominal": budget,
        "ra": policy.intra_mask,
        "re": policy.inter_mask,
        "keep_epochs": len(epochs),
        "num_epochs": num_epochs,
        "keep_patches": len(patches),
        "num_patches": num_patches,
        "epoch_keep": epochs,
        "epoch_standby": masked,
        "patch_keep": patches,
        "epoch_phase_span_train": phase_span(len(epochs), num_epochs),
        "patch_phase_span_train": phase_span(len(patches), num_patches),
        "deploy_phase": 0,
        "seconds_per_32epochs": seconds,
        "window_seconds": window_seconds,
        "realized_ratio": seconds / float(window_seconds),
        "count_variance": 0,
    }


def format_schedule(schedule: Dict) -> str:
    nominal = schedule["budget_nominal"]
    lines = [
        f"名义信号量 {nominal:.0%}    实际采集比例 {schedule['realized_ratio']:.6f}    "
        f"每 32 期固定采集 {schedule['seconds_per_32epochs']} / {schedule['window_seconds']} 秒",
        f"  时期间保留 {schedule['keep_epochs']}/{schedule['num_epochs']}    "
        f"训练相位种数 {schedule['epoch_phase_span_train']}    部署相位 {schedule['deploy_phase']}",
        f"  保留整期: {schedule['epoch_keep']}",
        f"  待机整期: {schedule['epoch_standby']}",
        f"  期内保留 {schedule['keep_patches']}/{schedule['num_patches']} 秒    "
        f"训练相位种数 {schedule['patch_phase_span_train']}",
        f"  保留秒:   {schedule['patch_keep']}",
        "  整晚 T 个 30s 期的采集秒数 = |{t : (t mod 32) 属于保留整期}| × 期内保留秒数，记录前即可算出。",
    ]
    return "\n".join(lines)


def assert_deployment_schedules() -> None:
    """Guard the timetable that the paper will cite."""
    expected = {
        1.00: {
            "epochs": list(range(32)),
            "masked": [],
            "patches": list(range(30)),
            "seconds": 960,
            "ratio": 1.0,
        },
        0.72: {
            "epochs": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30],
            "masked": [7, 15, 23, 31],
            "patches": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28],
            "seconds": 672,
            "ratio": 0.70,
        },
        0.40: {
            "epochs": [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30],
            "masked": [4, 9, 13, 18, 22, 27, 31],
            "patches": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "seconds": 375,
            "ratio": 0.390625,
        },
        0.10: {
            "epochs": list(range(0, 32, 2)),
            "masked": list(range(1, 32, 2)),
            "patches": [0, 5, 10, 15, 20, 25],
            "seconds": 96,
            "ratio": 0.10,
        },
    }
    for budget, exp in expected.items():
        got = deployment_schedule(budget)
        if got["epoch_keep"] != exp["epochs"] or got["epoch_standby"] != exp["masked"]:
            raise AssertionError(f"epoch timetable changed for {budget}: {got['epoch_keep']}")
        if got["patch_keep"] != exp["patches"]:
            raise AssertionError(f"patch timetable changed for {budget}: {got['patch_keep']}")
        if got["seconds_per_32epochs"] != exp["seconds"]:
            raise AssertionError(f"seconds changed for {budget}")
        if abs(got["realized_ratio"] - exp["ratio"]) > 1e-12:
            raise AssertionError(f"ratio changed for {budget}: {got['realized_ratio']}")
        _assert_equal_interval(got["epoch_keep"], 32)
        _assert_equal_interval(got["patch_keep"], 30)
        _assert_phase_counts(got["keep_epochs"], 32, got["epoch_phase_span_train"])
        _assert_phase_counts(got["keep_patches"], 30, got["patch_phase_span_train"])


def _assert_equal_interval(indices: List[int], num_tokens: int) -> None:
    if len(indices) >= num_tokens:
        return
    ordered = sorted(indices)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + num_tokens - ordered[-1])
    if max(gaps) - min(gaps) > 1:
        raise AssertionError(f"indices are not equal-interval: {ordered} gaps={gaps}")


def _assert_phase_counts(keep: int, num_tokens: int, span: int) -> None:
    for offset in range(span):
        idx = strided_indices(keep, num_tokens, offset)
        if int(torch.unique(idx).numel()) != keep:
            raise AssertionError(f"phase {offset} collapsed a keep-{keep}/{num_tokens} grid")
