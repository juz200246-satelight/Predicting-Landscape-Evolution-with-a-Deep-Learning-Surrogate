import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# change: Line12, 17, 33

# =========================
# folders
# =========================
os.makedirs("outputs_datasets", exist_ok=True)

# =========================
# load data
# =========================
all_z = np.load("datasets/all_z.npy")

print("Loaded terrain shape:", all_z.shape)

# shape = (N_sim, T, H, W)
num_simulations = all_z.shape[0]

# =====================================================
# loop over each simulation
# =====================================================
for sim_idx in range(num_simulations):

    print(f"\nProcessing simulation {sim_idx}")

    z_all = all_z[sim_idx]   # shape = (120,128,128)

    sim_output_dir = f"outputs_datasets/sim_{sim_idx:03d}"
    os.makedirs(sim_output_dir, exist_ok=True)

    # =====================================================
    # Part 1: make 2D GIF
    # =====================================================
    frames = []

    vmin = z_all.min()
    vmax = z_all.max()

    for t in range(z_all.shape[0]):

        fig, ax = plt.subplots(figsize=(5, 5))

        im = ax.imshow(
            z_all[t],
            cmap="terrain",
            vmin=vmin,
            vmax=vmax,
            origin="upper"
        )

        ax.set_title(f"Simulation {sim_idx} | Frame {t}")
        ax.axis("off")

        frame_path = f"{sim_output_dir}/frame_{t:03d}.png"

        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    gif_path = f"{sim_output_dir}/terrain_evolution.gif"

    imageio.mimsave(
        gif_path,
        frames,
        duration=0.25,
        loop=0
    )

    print(f"Saved GIF to: {gif_path}")

    # =====================================================
    # Part 2: make final 3D terrain
    # =====================================================
    z = z_all[-1]

    nrows, ncols = z.shape

    x = np.arange(ncols)
    y = np.arange(nrows)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        z,
        cmap="terrain",
        linewidth=0,
        antialiased=False,
        shade=True
    )

    # Keep original orientation
    ax.view_init(elev=45, azim=-90)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_box_aspect((1, 1, 0.25))

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    terrain_3d_path = f"{sim_output_dir}/terrain_3d.png"

    plt.savefig(
        terrain_3d_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="black"
    )

    plt.close(fig)

    print(f"Saved 3D terrain to: {terrain_3d_path}")

print("\nAll simulations processed!")