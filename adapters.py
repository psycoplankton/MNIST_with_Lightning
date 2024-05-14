from torch import Tensor, nn, flatten

class LinearAdapter(nn.Module):
    def __init__(self, in_features: int, out_features : int, flatten_input : bool = False) -> None:
        super().__init__()
        self.flatten_input : bool = flatten_input
        self.linear : Linear = nn.Linear(in_features, out_features)