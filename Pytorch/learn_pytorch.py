import wandb
import time
import torch

from Pytorch.environment_pytorch import WaterTank
from Pytorch.model_pytorch import log_weights_and_derivatives
from params import influx_params, env_params, model_params, run_params, start_params, optimizer_params


def learn_pytorch(name, model, optimizer, should_log=True):
    # timer
    timer = {
        "epoch": [],
        "step": [],
        "take_step": [],
        "parameter_update": [],
        "log": []
    }

    def record_time(section_name):
        start_time = time.time()
        timer[section_name].append(start_time)

    def update_timer(section_name):
        end_time = time.time()
        elapsed_time = end_time - timer[section_name][-1]
        timer[section_name][-1] = elapsed_time

    if should_log:
        wandb.init(
            project="Water Tank",
            name=name,
            config={
                "environment": env_params,
                "influx": influx_params,
                "model": model_params,
                "optimizer": optimizer_params,
                "run": run_params,
                "start": start_params
            }
        )

    seed = 42
    torch.manual_seed(seed)

    prev_weights = [param.detach().clone() for param in model.parameters()]

    # Training loop
    for epoch in range(run_params["num_epochs"]):
        record_time("epoch")
        water_tank = WaterTank(start_params, env_params, influx_params)

        for step in range(run_params["num_steps"]):
            record_time("step")
            record_time("take_step")
            state = water_tank.get_state()

            outflux_percentage = model(state)

            # Take a step in the environment
            reward = water_tank.step(outflux_percentage)

            update_timer("take_step")
            record_time("parameter_update")

            # Update the MLP using the reward and gradients
            optimizer.zero_grad()
            reward.backward()
            optimizer.step()

            update_timer("parameter_update")
            update_timer("step")

        if should_log:
            record_time("log")
            # Log the episode reward and other relevant information
            log_weights_and_derivatives(model, prev_weights)
            wandb.log(
                {
                    "loss": reward.item(),
                    "level": water_tank.level.item(),
                    "outflux percentage": outflux_percentage.item()
                }
            )
            update_timer("log")
        prev_weights = [param.detach().clone() for param in model.parameters()]
        update_timer("epoch")

    wandb.finish()
    return model, timer
