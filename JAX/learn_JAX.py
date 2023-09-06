import time
import optax
import wandb
import jax
import jax.numpy as jnp
from JAX.model_JAX import log_weights_and_derivatives_JAX
from params import influx_params, env_params, model_params, run_params, start_params, optimizer_params


def learn_JAX(name, weight_params, level, curr_time, optimizer, opt_state, take_step, should_log=True):
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

    def update_func(grads, opt_state, weight_params):
        updates, opt_state = optimizer.update(grads, opt_state, weight_params)
        weight_params = optax.apply_updates(weight_params, updates)
        return opt_state, weight_params

    update = jax.jit(update_func)

    prev_weights = weight_params

    # Training loop
    for epoch in range(run_params["num_epochs"]):
        record_time("epoch")
        cumm_loss = 0
        for step in range(run_params["num_steps"]):
            record_time("step")
            record_time("take_step")
            buf = take_step(weight_params, level, curr_time)
            (reward, aux_data), grads = buf
            level, curr_time, influx, outflux_percentage = aux_data
            update_timer("take_step")

            cumm_loss = cumm_loss + reward

            record_time("parameter_update")
            # updates, opt_state = optimizer.update(grads, opt_state, weight_params)
            # weight_params = optax.apply_updates(weight_params, updates)
            opt_state, weight_params = update(grads, opt_state, weight_params)
            update_timer("parameter_update")
            update_timer("step")

        if should_log:
            record_time("log")
            # Log the episode reward and other relevant information
            log_weights_and_derivatives_JAX(weight_params, prev_weights, grads)
            wandb.log(
                {
                    "loss": float(reward),
                    "level": float(level[0][0]),
                    "outflux percentage": float(outflux_percentage[0][0]),
                    "cummulative loss": float(cumm_loss)
                }
            )
            update_timer("log")

        prev_weights = weight_params

        # restart starting conditions
        level = jnp.array([[start_params["level"]]])
        curr_time = jnp.array([[start_params["time"]]])
        state = jnp.concatenate((level, curr_time), axis=1)

        update_timer("epoch")

    wandb.finish()
    return weight_params, timer
