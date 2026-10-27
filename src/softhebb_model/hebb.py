import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftHebbLinear(nn.Module):
  def __init__(
    self,
    in_features,
    out_features,
    *,
    plasticity=True,
    t_inverse=1.0,
    lr_hebb=0.01,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.t_inverse = t_inverse
    self.lr_hebb = lr_hebb
    self.plasticity = plasticity

    # 가중치 초기화 및 단위 구(Unit Sphere) 정규화
    self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
    self.normalize_weights()

  def train(self, mode=True):
    super().train(mode)
    self.plasticity = mode

  def forward(self, x):
    post_x = F.linear(x, self.weight)  
    if self.plasticity:
      self.update_hebbian(x, post_x)
  
    return post_x

  @torch.no_grad()
  def update_hebbian(self, x, post_x):
    wta = F.softmax(post_x * self.t_inverse, dim=1)  # (batch_size, out_features)

    yx = torch.matmul(wta.t(), x)  # Overlap between winner take all and inputs
    yu = torch.multiply(wta, post_x)
    yu = torch.sum(yu.t(), dim=1).unsqueeze(1)

    delta_weight = yx - yu.view(-1, 1) * self.weight

    self.weight.add_(self.lr_hebb * delta_weight)
    self.normalize_weights()

  @torch.no_grad()
  def normalize_weights(self):
    norm = self.weight.norm(p=2, dim=1, keepdim=True)
    self.weight.div_(norm.clamp(min=1e-12))