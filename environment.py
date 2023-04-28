import numpy as np
import torch


class WaterTank:
    def __init__(self, params):
        self.params = params["environment"]
        self.time_step = self.params["time_step"] if "time_step" in self.params.keys() else 0.1
        self.target_level = self.params["target_level"] if "target_level" in self.params.keys() else 10
        self.level = torch.tensor([self.params["initial_level"] if "initial_level" in self.params.keys() else 0],
                                  dtype=torch.float32)
        self.time = self.params["initial_time"] if "initial_time" in self.params.keys() else 0
        self.max_outflux = self.params["max_outflux"] if "max_outflux" in self.params.keys() else 1

        # history
        self.past_level = self.level
        self.past_time = [self.time]
        self.past_reward = torch.empty(0)
        self.past_influx = np.empty(0)
        self.past_outflux = torch.empty(0)

    def step(self, outflux_percentage, influx):
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


def influx_function(time, params):
    frequency = params["influx"]["frequency"]
    amplitude = params["influx"]["amplitude"]
    offset = params["influx"]["offset"]

    return amplitude * np.sin(2 * np.pi * frequency * time) + offset
