influx_params = {
    "amplitude": 0.25,
    "frequency": 0.1,
    "offset": 0.5
}

env_params = {
    "time_step": 0.1,
    "target_level": 10,
    "max_outflux": 1,
    "initial_level": 10,
    "initial_time": 0
}

# remove the first element for JAX
model_params = {
    "layer_sizes": [2, 32, 1]
}

run_params = {
    "num_epochs": 100,
    "num_steps": 500
}

start_params = {
    "level": 10,
    "time": 0
}

optimizer_params = {
    "learning_rate": 1e-3
}
