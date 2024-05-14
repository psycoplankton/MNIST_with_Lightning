from torch import Tensor, nn


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_fn(preds, target)