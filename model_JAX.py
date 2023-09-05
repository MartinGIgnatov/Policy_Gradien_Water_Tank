from typing import Sequence
import wandb
import jax
import flax
from flax import linen as nn
from jax import random
import jax.numpy as jnp


class MLP_Jax(nn.Module):
    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        # ~10% faster
        # x = nn.Dense(self.layers[0], name=f'layers_{0}')(x)
        # x = nn.relu(x)
        # x = nn.Dense(self.layers[1], name=f'layers_{1}')(x)
        # x = nn.sigmoid(x)
        
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, name=f'layers_{i}')(x)
            if i != len(self.layer_sizes) - 1:
                x = nn.relu(x)
                # x = nn.leaky_relu(x, negative_slope = 0.01)
            else:
                x = nn.sigmoid(x)
        return x


def log_weights_and_derivatives_JAX(weights, prev_weights, gradients):
    weights_flatten = jax.tree_util.tree_flatten(weights)[0]
    weights_flatten = jax.tree_map(jnp.ravel, weights_flatten)
    weights_flatten = jnp.concatenate(weights_flatten)
    weights_flatten = jnp.absolute(weights_flatten)

    prev_weights_flatten = jax.tree_util.tree_flatten(prev_weights)[0]
    prev_weights_flatten = jax.tree_map(jnp.ravel, prev_weights_flatten)
    prev_weights_flatten = jnp.concatenate(prev_weights_flatten)
    prev_weights_flatten = jnp.absolute(prev_weights_flatten)

    gradients_flatten = jax.tree_util.tree_flatten(gradients)[0]
    gradients_flatten = jax.tree_map(jnp.ravel, gradients_flatten)
    gradients_flatten = jnp.concatenate(gradients_flatten)
    gradients_flatten = jnp.absolute(gradients_flatten)

    # Calculate weight changes
    weight_changes = jnp.abs(weights_flatten-prev_weights_flatten)

    max_abs_gradient = float(jnp.max(gradients_flatten))
    mean_abs_gradient = float(jnp.mean(gradients_flatten))

    max_abs_weight = float(jnp.max(weights_flatten))
    mean_abs_weight = float(jnp.mean(weights_flatten))

    max_abs_weight_change = float(jnp.max(weight_changes))
    mean_abs_weight_change = float(jnp.mean(weight_changes))

    wandb.log(
        {
            "max_abs_gradient": max_abs_gradient,
            "mean_abs_gradient": mean_abs_gradient,
            "max_abs_weight": max_abs_weight,
            "mean_abs_weight": mean_abs_weight,
            "max_abs_weight_change": max_abs_weight_change,
            "mean_abs_weight_change": mean_abs_weight_change
        }
    )


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
