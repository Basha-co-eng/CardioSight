import numpy as np
import torch
import torch.nn as nn
from backend.inference.predictor import ResNet1D34
from backend.utils.constants import N_MC_PASSES, MC_DROPOUT_P


class ResNet1D34_MCDropout(ResNet1D34):
    """
    ResNet1D34 with one Dropout layer before the final classification layer.
    The original federated checkpoints were trained without this Dropout layer.
    Since Dropout has no learnable parameters, the original checkpoint can still
    be loaded using strict=False.
    """
    def __init__(
        self,
        in_channels=12,
        num_classes=12,
        dropout_p=MC_DROPOUT_P
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes
        )
        self.mc_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.mc_dropout(x)
        x = self.fc(x)
        return x


def enable_mc_dropout(model: torch.nn.Module):
    """
    Keep the entire model in evaluation mode so that BatchNorm uses learned running statistics.
    Only Dropout layers are switched back to training mode so stochastic predictions are generated.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def compute_mc_dropout_uncertainty(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: torch.device,
    n_passes: int = N_MC_PASSES
):
    """
    Run multiple stochastic forward passes for one ECG.
    Returns:
        mean_prediction: shape (12,)
        uncertainty: shape (12,)
    """
    enable_mc_dropout(model)
    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(signal_tensor)
            probabilities = torch.sigmoid(logits)
            predictions.append(probabilities.cpu().numpy()[0])

    predictions = np.stack(predictions, axis=0)
    mean_prediction = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    return mean_prediction, uncertainty
