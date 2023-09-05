import torch
import wandb
from torch import nn


class MLP(nn.Module):
    def __init__(self, model_params):
        super(MLP, self).__init__()

        # self.params = params["model"]
        # self.fc1 = nn.Linear(self.params["input_size"], self.params["hidden_size"])
        # self.fc2 = nn.Linear(self.params["hidden_size"], self.params["output_size"])
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

        self.layer_sizes = model_params["layer_sizes"]
        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            self.layers.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))
            if i == len(self.layer_sizes) - 1:
                self.layers.append(nn.Sigmoid())
            else:
                self.layers.append(nn.LeakyReLU()) # self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))

        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


def log_weights_and_derivatives(model, prev_weights):
    max_abs_gradient = 0
    mean_abs_gradient = 0
    count_gradient = 0

    max_abs_weight = 0
    mean_abs_weight = 0
    count_weight = 0

    # Calculate weight changes
    weight_changes = [torch.abs(curr - prev) for curr, prev in zip(model.parameters(), prev_weights)]

    max_abs_weight_changes = []
    mean_abs_weight_changes = []

    for name, param in model.named_parameters():
        grad = param.grad

        if grad is not None:
            max_abs_gradient = max(max_abs_gradient, torch.max(torch.abs(grad)))
            mean_abs_gradient += torch.mean(torch.abs(grad))
            count_gradient += 1

    if count_gradient > 0:
        mean_abs_gradient /= count_gradient

    for name, param in model.named_parameters():
        max_abs_weight = max(max_abs_weight, torch.max(torch.abs(param)))
        mean_abs_weight += torch.mean(torch.abs(param))
        count_weight += 1

    if count_weight > 0:
        mean_abs_weight /= count_weight

    for wc in weight_changes:
        max_abs_weight_changes.append(wc.max().item())
        mean_abs_weight_changes.append(torch.mean(wc).item())

    max_abs_weight_change = max(max_abs_weight_changes)
    mean_abs_weight_change = sum(mean_abs_weight_changes) / len(mean_abs_weight_changes)

    wandb.log(
        {
            "max_abs_gradient": max_abs_gradient,
            "mean_abs_gradient": mean_abs_gradient.item(),
            "max_abs_weight": max_abs_weight.item(),
            "mean_abs_weight": mean_abs_weight.item(),
            "max_abs_weight_change": max_abs_weight_change,
            "mean_abs_weight_change": mean_abs_weight_change
        }
    )
