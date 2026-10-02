from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class SignClassifier(nn.Module):
    INPUT_FORMAT = "image"
    TRAINABLE_LAYERS = ("layer4", "fc")

    def __init__(
        self,
        class_count: int,
        pretrained: bool = True,
        hidden_size: int = 128,
        dropout: float = 0.5,
        trainable_layers: tuple[str, ...] = TRAINABLE_LAYERS,
    ) -> None:
        super().__init__()
        self.trainable_layers = tuple(trainable_layers)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, class_count),
        )
        self._freeze_initial_layers()
        self.train()

    # Público
    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)

    def train(self, mode: bool = True):
        super().train(mode)

        for name, layer in self.network.named_children():
            if name not in self.trainable_layers:
                layer.eval()
        return self

    # Privado
    def _freeze_initial_layers(self) -> None:
        available_layers = dict(self.network.named_children())
        unknown_layers = set(self.trainable_layers) - available_layers.keys()
        if unknown_layers:
            raise ValueError(
                f"Camadas desconhecidas: {', '.join(sorted(unknown_layers))}. "
                f"Disponíveis: {', '.join(available_layers)}."
            )

        self.network.requires_grad_(False)
        for name in self.trainable_layers:
            getattr(self.network, name).requires_grad_(True)
