import jax
import jax.numpy as jnp
from functools import partial
from model_JAX import MLP_Jax
from jax import make_jaxpr
import orbax
from flax.training import orbax_utils


def calc_influx(time, influx_params):
    frequency = influx_params["frequency"]
    amplitude = influx_params["amplitude"]
    offset = influx_params["offset"]
    influx = amplitude * jnp.sin(2 * jnp.pi * frequency * time) + offset
    return influx


def _take_step(weight_params, level, time, env_params, influx_params, model_params):
    max_outflux = env_params["max_outflux"]
    time_step = env_params["time_step"]
    target_level = env_params["target_level"]
    model = MLP_Jax(model_params["layer_sizes"][1:])

    influx = calc_influx(time, influx_params)
    state = jnp.concatenate((level, influx), axis=1)
    outflux_percentage = model.apply(weight_params, state)

    level = (influx - outflux_percentage * max_outflux) * time_step + level
    # assures level is min 0 but keeps the backprop
    level = (jax.lax.stop_gradient(level) < 0) * jax.lax.stop_gradient(level) * (-1) + level
    time = time_step + time

    loss = jnp.mean(jnp.abs(level - target_level))

    return loss, (level, time, influx, outflux_percentage)


class WaterTank_Jax:
    def __init__(self, env_params, influx_params, model_params):
        self.env_params = env_params
        self.influx_params = influx_params
        self.model_params = model_params

        # history
        self.past_level = [env_params["initial_level"]]
        self.past_time = [env_params["initial_time"]]
        self.past_reward = []
        self.past_influx = []
        self.past_outflux = []

    def get_influx_func(self):
        return jax.jit(partial(calc_influx, influx_params=self.influx_params))

    def get_take_step(self):
        return jax.jit(
            jax.value_and_grad(
                partial(
                    _take_step,
                    env_params=self.env_params,
                    influx_params=self.influx_params,
                    model_params=self.model_params
                ),
                has_aux=True
            )
        )

    def store_step(self, level, time, reward, influx, outflux):
        self.past_level.append(float(level[0][0]))
        self.past_time.append(float(time[0][0]))
        self.past_reward.append(float(reward))
        self.past_influx.append(float(influx[0][0]))
        self.past_outflux.append(float(outflux[0][0]))

    def get_history(self):
        return self.past_level, self.past_time, self.past_reward, self.past_influx, self.past_outflux


if __name__ == "__main__":
    from params import influx_params, env_params, model_params, run_params, start_params
    from jax import random

    level = jnp.array([[start_params["level"]]])
    time = jnp.array([[start_params["time"]]])
    state = jnp.concatenate((level, time), axis=1)

    model = MLP_Jax(model_params["layers"][1:])  # have to remove the first element
    weight_params = model.init(random.PRNGKey(0), state)
    # print(_take_step(weight_params, level, time, env_params, influx_params, model_params))

    a = WaterTank_Jax(env_params, influx_params, model_params)

    res = a.get_take_step()(weight_params, level, time)
    (reward, (level, time, influx, outflux_percentage)), weight_grad = res
    print(reward, level, time, influx, outflux_percentage, weight_grad, end="/n")

    # print(make_jaxpr(a.get_take_step())(weight_params, level, time))
