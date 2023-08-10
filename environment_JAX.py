import jax
import jax.numpy as jnp
from functools import partial
from model_JAX import MLP_Jax
from jax import make_jaxpr


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
    model = MLP_Jax(model_params["layers"][1:])

    state = jnp.concatenate((level, time), axis=1)

    influx = calc_influx(time, influx_params)
    outflux_percentage = model.apply(weight_params, state)

    level = (influx - outflux_percentage * max_outflux) * time_step + level
    # assures level is min 0 but keeps the backprop
    level = (level < 0) * jax.lax.stop_gradient(level) * (-1) + level
    time = time_step + time

    reward = jnp.abs(level - target_level)  # loss

    return reward, level, time, influx, outflux_percentage


class WaterTank_Jax:
    def __init__(self, env_params, influx_params, model_params):
        self.env_params = env_params
        self.influx_params = influx_params
        self.model_params = model_params

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


if __name__ == "__main__":
    env_params = {
        "time_step": 0.1,
        "target_level": 10,
        "initial_level": 10,
        "initial_time": 0,
        "max_outflux": 1,
    }
    influx_params = {
        "amplitude": 0.25,
        "frequency": 0.1,
        "offset": 0.5
    }

    a = WaterTank_Jax(env_params, influx_params)

    print(make_jaxpr(a.get_step())(0, 0, 0, 0))
