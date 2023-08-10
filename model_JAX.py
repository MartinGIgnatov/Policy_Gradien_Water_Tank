from typing import Sequence
import wandb
import jax
import flax
from flax import linen as nn
from jax import random
import jax.numpy as jnp


class MLP_Jax(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, name=f'layers_{i}')(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
            else:
                x = nn.sigmoid(x)
        return x


# todo
# def log_weights_and_derivatives(params, prev_params):
#     max_abs_gradient = 0
#     mean_abs_gradient = 0
#     count_gradient = 0
#
#     max_abs_weight = 0
#     mean_abs_weight = 0
#     count_weight = 0
#
#     # Calculate weight changes
#     weight_changes = []
#
#     max_abs_weight_changes = []
#     mean_abs_weight_changes = []
#
#     for name, param in model.named_parameters():
#         grad = param.grad
#
#         if grad is not None:
#             max_abs_gradient = max(max_abs_gradient, torch.max(torch.abs(grad)))
#             mean_abs_gradient += torch.mean(torch.abs(grad))
#             count_gradient += 1
#
#     if count_gradient > 0:
#         mean_abs_gradient /= count_gradient
#
#     for name, param in model.named_parameters():
#         max_abs_weight = max(max_abs_weight, torch.max(torch.abs(param)))
#         mean_abs_weight += torch.mean(torch.abs(param))
#         count_weight += 1
#
#     if count_weight > 0:
#         mean_abs_weight /= count_weight
#
#     for wc in weight_changes:
#         max_abs_weight_changes.append(wc.max().item())
#         mean_abs_weight_changes.append(torch.mean(wc).item())
#
#     max_abs_weight_change = max(max_abs_weight_changes)
#     mean_abs_weight_change = sum(mean_abs_weight_changes) / len(mean_abs_weight_changes)
#
#     wandb.log(
#         {
#             "max_abs_gradient": max_abs_gradient,
#             "mean_abs_gradient": mean_abs_gradient.item(),
#             "max_abs_weight": max_abs_weight.item(),
#             "mean_abs_weight": mean_abs_weight.item(),
#             "max_abs_weight_change": max_abs_weight_change,
#             "mean_abs_weight_change": mean_abs_weight_change
#         }
#     )


if __name__ == "__main__":
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (3, 4))
    print("shape x", x.shape)

    model = MLP_Jax(layers=[3, 4, 5])
    params = model.init(key2, x)
    y = model.apply(params, x)

    print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, params))
    print('output:\n', y)

    list_of_lists = [
        {'a': 3},
        [1, 2, 3],
        [1, [1, 2]],
        [1, 2, 3, 4]
    ]

    print(jax.tree_util.tree_flatten(list_of_lists)[0])
