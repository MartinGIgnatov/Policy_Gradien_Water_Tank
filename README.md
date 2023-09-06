# Policy_Gradien_Water_Tank
A Reinforcement Learning algorithm for optimizing the controls of a water tank. The repo uses both PyTorch and JAX to perform the same action and compare their performance.

# Environment 
The water tank can be considered to hold an infinite amount of water but we want to have a constant pre-decided amount inside. 
It is looked at in a chronological fashion by having separate steps in one episode. 
At each step, we have some influx and some outflux, so the volume inside the tank changes.
A model has to control the amount of outflux of the water tank, knowing the current volume and the following influx.

# Start Up
The project has two main files. One for PyTorch and one for JAX, both should be directly runnable.
One can control the parameters of the whole system from the parameter file.
Each model has its parameters saved, however, they are overwritten every time a new model is trained.
