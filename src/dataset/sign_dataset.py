import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        processor,
        class_to_index: dict[str, int],
        training: bool = False,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.class_to_index = class_to_index
        self.training = training

    # Público
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        # O mesmo processamento determinístico é usado em todas as divisões.
        inputs = self.processor.process(row["sample_id"])

        label = self.class_to_index[row["class_id"]]
        return torch.from_numpy(inputs), label
