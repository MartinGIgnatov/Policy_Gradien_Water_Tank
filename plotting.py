import matplotlib.pyplot as plt


def plot_history(past_level, past_time, past_reward, past_influx, past_outflux):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Plot the level with time on the first subplot
    ax[0].plot(past_time, past_level)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Level")
    ax[0].set_title("Level vs Time")
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    ax[0].grid(True)

    # Plot the reward with time on the second subplot
    ax[1].plot(past_time[1:], past_reward)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlim(left=0)
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("Loss vs Time")
    ax[1].grid(True)

    # Plot the influx and outflux with time
    ax[2].plot(past_time[1:], past_influx, label="Influx")
    ax[2].plot(past_time[1:], past_outflux, label="Outflux")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Flux")
    ax[2].set_title("Flux vs Time")
    ax[2].legend()
    ax[2].grid(True)

    # Display the figure
    plt.show()
