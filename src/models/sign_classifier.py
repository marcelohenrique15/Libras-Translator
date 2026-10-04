from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class SignClassifier(nn.Module):
    INPUT_FORMAT = "image"
    OUTPUT_FORMAT = "probabilities"
    TRAINABLE_LAYERS = ("layer4", "fc")

    def __init__(
        self,
        class_count: int,
        pretrained: bool = True,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        trainable_layers: tuple[str, ...] = TRAINABLE_LAYERS,
    ) -> None:
        super().__init__()
        if class_count < 2 or hidden_size < 1 or num_layers < 1:
            raise ValueError("class_count deve ser >= 2; hidden_size e num_layers devem ser >= 1.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout deve estar no intervalo [0, 1).")
        self.trainable_layers = tuple(trainable_layers)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        # A primeira camada mantém os mesmos nomes de pesos dos checkpoints anteriores.
        layers = [nn.BatchNorm1d(512)]
        input_size = 512
        for _ in range(num_layers):
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, class_count))
        self.network.fc = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        self._freeze_initial_layers()
        self.train()

    # Público
    def forward(self, images: Tensor) -> Tensor:
        return self.softmax(self.forward_logits(images))

    def forward_logits(self, images: Tensor) -> Tensor:
        # CrossEntropyLoss recebe logits para calcular log-softmax de forma estável.
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
