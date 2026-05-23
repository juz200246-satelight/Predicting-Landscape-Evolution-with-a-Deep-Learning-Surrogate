import os
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
)

# =========================
# output folder
# =========================
os.makedirs("datasets", exist_ok=True)

# =========================
# dataset parameters
# =========================
NUM_SIMULATIONS = 2
# NUM_SIMULATIONS = 270

N_FRAMES = 120
NROWS = 128
NCOLS = 128
DX = 10.0

STEPS_PER_FRAME = 20
DT = 50.0

# Use the same physical logic as generate_one_simulation.py
BASE_SLOPE_Y = 0.03
BASE_SLOPE_X = 0.01
BASE_NOISE_AMP = 1.0
BASE_K_SP = 5e-5
BASE_DIFFUSIVITY = 0.01
BASE_UPLIFT_RATE = 1e-3

# If True, each simulation has small variation while still keeping the same "mountain-to-sea" logic.
# If False, only the random noise seed changes.
USE_SMALL_VARIATION = True

all_z = []

for sim in range(NUM_SIMULATIONS):
    print(f"\n===== Simulation {sim + 1}/{NUM_SIMULATIONS} =====")

    rng = np.random.default_rng(sim)

    # =================================================
    # create grid
    # =================================================
    mg = RasterModelGrid((NROWS, NCOLS), DX)
    z = mg.add_zeros("topographic__elevation", at="node")

    x = mg.x_of_node
    y = mg.y_of_node

    # =================================================
    # terrain initialization
    # =================================================
    # IMPORTANT:
    # Use POSITIVE slope_y, same as generate_one_simulation.py.
    # Bottom boundary is open, so bottom should be lower and top should be higher.
    # This creates a proper mountain-to-sea drainage direction.
    if USE_SMALL_VARIATION:
        slope_y = BASE_SLOPE_Y * rng.uniform(0.85, 1.15)
        slope_x = BASE_SLOPE_X * rng.uniform(0.5, 1.5)
        noise_amp = BASE_NOISE_AMP * rng.uniform(0.8, 1.3)

        K_sp = BASE_K_SP * rng.uniform(0.8, 1.2)
        diffusivity = BASE_DIFFUSIVITY * rng.uniform(0.8, 1.2)
        uplift_rate = BASE_UPLIFT_RATE * rng.uniform(0.8, 1.2)
    else:
        slope_y = BASE_SLOPE_Y
        slope_x = BASE_SLOPE_X
        noise_amp = BASE_NOISE_AMP

        K_sp = BASE_K_SP
        diffusivity = BASE_DIFFUSIVITY
        uplift_rate = BASE_UPLIFT_RATE

    z[:] = (
        slope_y * y
        + slope_x * x
        + rng.random(mg.number_of_nodes) * noise_amp
    )

    # =================================================
    # boundary condition
    # =================================================
    # Same as generate_one_simulation.py:
    # close right/top/left, open bottom as outlet.
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=True,
        left_is_closed=True,
        bottom_is_closed=False,
    )

    # =================================================
    # Landlab components
    # =================================================
    fa = FlowAccumulator(mg, flow_director="D8")

    sp = FastscapeEroder(
        mg,
        K_sp=K_sp,
        m_sp=0.5,
        n_sp=1.0,
    )

    ld = LinearDiffuser(
        mg,
        linear_diffusivity=diffusivity,
    )

    # =================================================
    # simulation loop
    # =================================================
    frames = []

    for frame in range(N_FRAMES):
        for step in range(STEPS_PER_FRAME):
            # tectonic uplift on core nodes
            z[mg.core_nodes] += uplift_rate * DT

            # fluvial incision
            fa.run_one_step()
            sp.run_one_step(DT)

            # hillslope diffusion
            ld.run_one_step(DT)

        z_grid = z.reshape((NROWS, NCOLS)).copy()
        frames.append(z_grid)

        if frame % 20 == 0 or frame == N_FRAMES - 1:
            print(
                f"sim={sim:03d}, frame={frame:03d}, "
                f"min={z_grid.min():.3f}, max={z_grid.max():.3f}, "
                f"range={z_grid.max() - z_grid.min():.3f}"
            )

    frames = np.array(frames, dtype=np.float32)
    print("simulation shape:", frames.shape)

    all_z.append(frames)

# =====================================================
# save final dataset
# =====================================================
all_z = np.array(all_z, dtype=np.float32)

print("\nFinal dataset shape:")
print(all_z.shape)

save_path = "datasets/all_z.npy"
np.save(save_path, all_z)

print(f"\nSaved dataset to: {save_path}")
