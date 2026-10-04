from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.utils.constants import FEDAVG_MODEL_PATH, FEDPROX_MODEL_PATH, FEDOPT_MODEL_PATH


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet1D34(nn.Module):
    def __init__(self, in_channels=12, num_classes=12):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.layer1 = self._make_layer(
            64, 64, num_blocks=3, stride=1
        )

        self.layer2 = self._make_layer(
            64, 128, num_blocks=4, stride=2
        )

        self.layer3 = self._make_layer(
            128, 256, num_blocks=6, stride=2
        )

        self.layer4 = self._make_layer(
            256, 512, num_blocks=3, stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def _make_layer(
        self,
        in_channels,
        out_channels,
        num_blocks,
        stride
    ):
        layers = [
            ResidualBlock1D(
                in_channels,
                out_channels,
                stride
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock1D(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)

        return x


def load_model_from_path(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = ResNet1D34(in_channels=12, num_classes=12)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_fedavg_model(device: torch.device):
    return load_model_from_path(FEDAVG_MODEL_PATH, device)


def load_fedprox_model(device: torch.device):
    return load_model_from_path(FEDPROX_MODEL_PATH, device)


def load_fedopt_model(device: torch.device):
    return load_model_from_path(FEDOPT_MODEL_PATH, device)


def predict(model, signal, device):
    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(signal_tensor)
        probabilities = torch.sigmoid(logits)

    return (
        logits.cpu().numpy()[0],
        probabilities.cpu().numpy()[0]
    )
