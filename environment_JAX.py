import jax
import jax.numpy as jnp
from functools import partial

from jax import make_jaxpr


class WaterTank_Jax:
    def __init__(self, params=None):
        new_params = {
            "time_step": 0.1,
            "target_level": 10,
            "initial_level": 10,
            "initial_time": 0,
            "max_outflux": 1,
            "amplitude": 0.25,
            "frequency": 0.1,
            "offset": 0.5
        }

        if params is None:
            params = {"environment": new_params}
        elif "environment" in params:
            for key in new_params:
                params["environment"][key] = params["environment"].get(key, new_params[key])
        else:
            params["environment"] = new_params

        self.params = params["environment"]
        self.whole_params = params

    def get_params(self):
        return self.whole_params

    def get_step(self):
        return jax.jit(partial(self._step, params=self.params))

    def _step(self, state, params):
        frequency = params["frequency"]
        amplitude = params["amplitude"]
        offset = params["offset"]

        influx = amplitude * jnp.sin(2 * jnp.pi * frequency * state["time"]) + offset

        state["level"] = (influx - state["outflux_percentage"] * params["max_outflux"]) * params["time_step"] + state[
            "level"]
        state["level"] = (state["level"] < 0) * jax.lax.stop_gradient(state["level"]) * (-1) + state[
            "level"]  # assures level is min 0 but keeps the backprop
        state["time"] += params["time_step"]
        reward = jnp.abs(state["level"] - params["target_level"])  # loss

        return state, reward


if __name__ == "__main__":
    params = {
        "environment": {
            "time_step": 0.1,
            "target_level": 10,
            "initial_level": 10,
            "initial_time": 0,
            "max_outflux": 1,
        }
    }
    a = WaterTank_Jax(params)

    state = {
        "level": 0,
        "outflux_percentage": 0,
        "time": 0
    }

    print(make_jaxpr(a.get_step())(state))
