from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812 -- such is the convention for torch.

if TYPE_CHECKING:
    from pathlib import Path

# queries per distance pass: each takes queries x reference windows floats per channel
CHUNK = 512


class MatrixProfile:
    def __init__(self, *, n_targets: int, normalize: bool, device: str) -> None:
        self._n_targets = n_targets
        self._normalize = normalize
        self._device = device
        self._reference: torch.Tensor | None = None  # (n_targets, windows, m)
        self._norms: torch.Tensor | None = None  # (n_targets, windows), squared

    def _set(self, reference: torch.Tensor | None) -> None:
        self._reference = reference
        self._norms = None if reference is None else reference.square().sum(-1)

    def _subsequences(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, m, n_features) -> (n_targets, B, m), z-normalized if `normalize`."""
        x = windows[:, :, : self._n_targets].to(self._device).permute(2, 0, 1)
        if not self._normalize:
            return x
        centred = x - x.mean(-1, keepdim=True)
        std = centred.square().mean(-1, keepdim=True).sqrt()
        # exact, as stumpy's rolling_isconstant
        constant = x.amax(-1, keepdim=True) == x.amin(-1, keepdim=True)
        return torch.where(constant, 0.0, centred / std)

    def outputs(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(distance to the nearest reference window, 0): a window identical to one
        in the reference scores 0.
        """
        distances = self.predict(windows)
        return distances, torch.zeros_like(distances)

    @torch.no_grad()
    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, window_size, n_features) -> (B, 1, n_targets)."""
        return torch.cat([self._nearest(chunk) for chunk in windows.split(CHUNK)])

    def _nearest(self, windows: torch.Tensor) -> torch.Tensor:
        queries = self._subsequences(windows)
        # squared distances up to each query's own norm: the minimum is taken before
        # the root, so only one value per query and channel is completed
        nearest = [
            torch.addmm(norms, q, r.T, alpha=-2).amin(1)
            for q, r, norms in zip(queries, self._reference, self._norms, strict=True)
        ]
        squared = torch.stack(nearest, dim=1) + queries.square().sum(-1).T
        return squared.clamp_min(0).sqrt()[:, None].cpu()

    def fit(self, windows: torch.Tensor) -> None:
        """Makes `windows` the reference, replacing the previous one."""
        self._set(self._subsequences(windows))

    def loss(self, windows: torch.Tensor) -> float:
        """Mean squared distance to the nearest reference window."""
        return F.mse_loss(*self.outputs(windows)).item()

    def state(self) -> torch.Tensor | None:
        return self._reference  # never changed in place, so no copy

    def restore(self, state: torch.Tensor | None) -> None:
        self._set(state)

    def save(self, path: Path) -> None:
        torch.save(self._reference, path)

    def load(self, path: Path) -> None:
        self._set(torch.load(path, map_location=self._device))
