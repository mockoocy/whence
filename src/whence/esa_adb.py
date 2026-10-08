# Loads ESA-ADB dataset from the parquet file (after my non-standard preprocessing).
# We apply a train/test split or however one may call it.
# To better showcase the "streaming" use case I will stream the
# parquet file, keeping track only of subset of the data.

from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple, Self

import polars as pl
import torch
from pyarrow.parquet import ParquetFile

from whence.stream import Stream

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from datetime import datetime
    from pathlib import Path


class Sample(NamedTuple):
    data: torch.Tensor  # (sample_len, n_features)
    labels: torch.Tensor  # (sample_len, n_targets) label codes


def sliding_windows(
    chunks: Iterable[torch.Tensor],
    length: int,
) -> Iterator[torch.Tensor]:
    """Transform tensor into another tensor containing sliding windows with stride=1"""
    rest = None
    for chunk in chunks:
        buffer = chunk if rest is None else torch.cat([rest, chunk])
        if len(buffer) < length:
            rest = buffer
            continue
        yield from buffer.unfold(0, length, 1).transpose(1, 2)
        rest = buffer[len(buffer) - length + 1 :]


@dataclass(frozen=True)
class ColumnSet:
    channels: tuple[str, ...]
    telecommands: tuple[
        str,
        ...,
    ]  # not neceserarilly used, only as input, never as targtet.
    labels: tuple[str, ...]  # per-channel labels

    @property
    def all(self) -> tuple[str, ...]:
        return self.channels + self.telecommands + self.labels

    @property
    def features(self) -> tuple[str, ...]:
        return self.channels + self.telecommands

    @classmethod
    def validation_gated_cl(cls: type[Self]) -> Self:
        channel_ids = range(41, 47)

        channels = tuple(f"channel_{i}" for i in channel_ids)
        labels = tuple(f"is_anomaly_channel_{i}" for i in channel_ids)

        return cls(
            channels=channels,
            telecommands=(),
            labels=labels,
        )

    @classmethod
    def targets(cls: type[Self], channels_csv: Path) -> Self:
        """Every annotated channel (Target=YES), in natural order."""
        meta = pl.read_csv(channels_csv).filter(pl.col("Target") == "YES")
        channels = tuple(
            sorted(meta["Channel"], key=lambda name: int(name.rsplit("_", 1)[1])),
        )
        return cls(
            channels=channels,
            telecommands=(),
            labels=tuple(f"is_anomaly_{c}" for c in channels),
        )


@cache
def row_at(path: Path, when: datetime) -> int:
    """Index of the first row at or after `when`; the file is sorted by timestamp."""
    hit = (
        pl.scan_parquet(path)
        .with_row_index()
        .filter(pl.col("timestamp") >= when)
        .select("index")
        .head(1)
        .collect(background=False)
    )
    # should be enforced be collect(background=False) type checker doesnt get it :/
    assert isinstance(hit, pl.DataFrame)
    if hit.is_empty():
        msg = f"no row at or after {when} in {path}"
        raise ValueError(msg)
    return hit.item()


class EsaAdbDataset:
    def __init__(
        self,
        path: Path,
        *,
        cols: ColumnSet,
        rows: slice = slice(None),
        read_batch_size: int = 512,
    ) -> None:
        assert rows.step in (None, 1), "rows must be contiguous"
        # a negative bound would silently count from the end of the file
        if (rows.start or 0) < 0 or (rows.stop or 0) < 0:
            msg = f"negative row bound in {rows}"
            raise ValueError(msg)
        self._cols = cols
        self._path = path
        # clamped to the file like a list slice, so open ends need no row count
        self._rows = range(ParquetFile(path).metadata.num_rows)[rows]
        self._read_batch_size = read_batch_size

    def __iter__(self) -> Generator[torch.Tensor]:
        file = ParquetFile(self._path)
        rows = self._rows
        # A parquet file is a sequence of independently readable row groups, whose
        # sizes are stored in it's metadata.
        # We can't really start reading from a particular row, so we will need to
        # "dummy read" some, to limit these reads row_groups are filtered beforehand.
        sizes = [
            file.metadata.row_group(group_index).num_rows
            for group_index in range(file.num_row_groups)
        ]
        firsts = list(accumulate(sizes, initial=0))
        groups = [
            group_index
            for group_index in range(len(sizes))
            if firsts[group_index] < rows.stop and rows.start < firsts[group_index + 1]
        ]

        position = firsts[groups[0]]  # index of the next batch's first row
        for batch in file.iter_batches(
            batch_size=self._read_batch_size,
            row_groups=groups,
            columns=self._cols.all,
        ):
            first, position = position, position + len(batch)
            if position <= rows.start:
                continue
            start = max(first, rows.start)
            df = pl.from_arrow(batch.slice(start - first, rows.stop - start))
            assert isinstance(df, pl.DataFrame)
            tensor = df.select(self._cols.all).to_torch(dtype=pl.Float32)
            assert isinstance(tensor, torch.Tensor)
            yield tensor
            if position >= rows.stop:
                break

    def stream(self, length: int, *, limit: int | None = None) -> Stream[Sample]:
        """The first `limit` sliding windows of `length` rows, split into feature
        and label parts (rows are laid out as `self._cols.all`).
        """
        n_features = len(self._cols.features)
        n_windows = max(0, len(self._rows) - length + 1)
        return (
            Stream(sliding_windows(self, length), n_windows)
            .map(lambda w: Sample(w[:, :n_features], w[:, n_features:]))
            .take(limit)
        )
