from torch import Tensor, nn

class Simple_model(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone : nn.Module = backbone
        self.adapter : nn.Module = adapter
        self.head : nn.Module = head

    def forward(self, x : Tensor) -> Tensor:
        x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.adapter(x)
        x = self.head(x)
        return x