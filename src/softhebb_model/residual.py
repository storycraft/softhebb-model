from torch import nn

class Residual(nn.Module):
  def __init__(self, inner: nn.Module):
    super().__init__()
    self.inner = inner

  def forward(self, x):
    return self.inner(x) + x
