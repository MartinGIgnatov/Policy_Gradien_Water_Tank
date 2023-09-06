import numpy as np
import torch


class WaterTank:
    def __init__(self, start_params, env_params, influx_params):
        self.env_params = env_params
        self.influx_params = influx_params
        self.start_params = start_params
        self.time_step = self.env_params["time_step"] if "time_step" in self.env_params.keys() else 0.1
        self.target_level = self.env_params["target_level"] if "target_level" in self.env_params.keys() else 10
        self.max_outflux = self.env_params["max_outflux"] if "max_outflux" in self.env_params.keys() else 1
        self.level = torch.tensor([self.start_params["level"] if "level" in self.start_params.keys() else 0],
            dtype=torch.float32)
        self.time = self.start_params["time"] if "time" in self.start_params.keys() else 0

        # history
        self.past_level = self.level
        self.past_time = [self.time]
        self.past_reward = torch.empty(0)
        self.past_influx = np.empty(0)
        self.past_outflux = torch.empty(0)

    def step(self, outflux_percentage):
        influx = self.influx_function()
        self.level = (influx - outflux_percentage * self.max_outflux) * self.time_step + self.level.detach().clone()
        self.level = (self.level < 0)*self.level.detach()*(-1) + self.level # assures level is min 0 but keeps the backprop
        self.time += self.time_step
        reward = self._calculate_loss()

        # Update the history by concatenating the new values
        self.past_level = torch.cat([self.past_level, self.level.detach().clone()])
        self.past_reward = torch.cat([self.past_reward, reward.detach().clone()])
        self.past_time.append(self.time)
        self.past_influx = np.append(self.past_influx, influx)
        self.past_outflux = torch.cat([self.past_outflux, outflux_percentage.detach().clone()])
        return reward

    def _calculate_loss(self):
        return torch.abs(self.level - self.target_level)

    def get_history(self):
        return self.past_level, self.past_time, self.past_reward, self.past_influx, self.past_outflux

    def get_state(self):
        return torch.tensor([self.level, self.influx_function()], dtype=torch.float32)

    def influx_function(self):
        frequency = self.influx_params["frequency"]
        amplitude = self.influx_params["amplitude"]
        offset = self.influx_params["offset"]

        return amplitude * np.sin(2 * np.pi * frequency * self.time) + offset
