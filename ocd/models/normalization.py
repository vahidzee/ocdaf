import torch

class TanhTransform(torch.nn.Module):
    def __init__(
        self,
        pre_act_scale: float,
        post_act_scale: float,
    ):
        super().__init__()
        self.pre_act_scale, self.post_act_scale = pre_act_scale, post_act_scale
        self.activation = torch.nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.pre_act_scale
        outputs = self.activation(outputs)
        return outputs * self.post_act_scale

    def extra_repr(self) -> str:
        pre_act = "x"
        pre_act = f"{pre_act} * {self.pre_act_scale}" if self.pre_act_scale != 1 else f"{pre_act}"
        act_name = "tanh"
        act = f"{act_name}({pre_act})"
        post_act = f"{act} * {self.post_act_scale}" if self.post_act_scale != 1 else f"{act}"
        return super().extra_repr() + f"forward: {post_act}"