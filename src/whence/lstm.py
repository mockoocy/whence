import torch
from torch import nn


class LstmForecasterModule(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: tuple[int, int],
        dropout: float = 0.5,
        horizon: int = 10,
        n_target: int | None = None,
    ) -> None:
        super().__init__()
        # number of output channels, may differ from input, e.g. when including
        # telecommands in the input, we do not really want to predict them.
        self._horizon = horizon
        self._n_target = n_target or n_features
        # nn.Sequential cannot hold an LSTM: it returns (output, (h, c)) and the
        # next module receives that tuple. The layers are wired by hand instead.
        self._lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size[0],
            batch_first=True,
        )
        self._dropout1 = nn.Dropout(p=dropout)
        self._lstm2 = nn.LSTM(
            input_size=hidden_size[0],
            hidden_size=hidden_size[1],
            batch_first=True,
        )
        self._dropout2 = nn.Dropout(p=dropout)
        self._head = nn.Linear(
            in_features=hidden_size[1],
            out_features=horizon * self._n_target,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(Batch, SequenceLength, Features) -> (Batch, horizon, n_target)"""
        out, _ = self._lstm1(x)
        out, _ = self._lstm2(self._dropout1(out))
        # only the final timestep's hidden state feeds the head
        out = self._head(self._dropout2(out[:, -1, :]))
        return out.view(-1, self._horizon, self._n_target)
