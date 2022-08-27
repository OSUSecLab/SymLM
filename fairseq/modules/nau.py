import torch
import math

torch.manual_seed(0)


class ReRegualizedLinearNACLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def forward(self, input, reuse=False):
        W = torch.clamp(self.W, -1, 1)
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
