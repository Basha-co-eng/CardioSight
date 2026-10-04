import numpy as np
import torch
import torch.nn.functional as F


class GradCAM1D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        target_class_idx: int,
        signal_length: int
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        score = output[0, target_class_idx]
        score.backward()

        weights = self.gradients.mean(
            dim=2,
            keepdim=True
        )

        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(
            cam,
            size=signal_length,
            mode="linear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam(
    model: torch.nn.Module,
    signal: np.ndarray,
    predicted_class_idx: int,
    device: torch.device
) -> np.ndarray:
    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    cam_engine = GradCAM1D(
        model,
        target_layer=model.layer4
    )

    heatmap = cam_engine.generate(
        signal_tensor,
        target_class_idx=predicted_class_idx,
        signal_length=signal.shape[-1]
    )

    return heatmap
