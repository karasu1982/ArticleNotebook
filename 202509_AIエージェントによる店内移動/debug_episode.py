# ==============================================
# File: debug_episode.py
# Print step-by-step transitions with a simple greedy policy
# ==============================================

import time
import numpy as np
from sari_sandbox import SariSandboxEnv, TaskConfig, ACTION_NAMES
from sari_sandbox import make_env

RENDER_DELAY = 0.0  # set to 0.1~0.3 for slow motion


def nearest_target(env: SariSandboxEnv):
    """Return the nearest (y,x) target tile: any remaining SKU tile if exists, else goal."""
    ay, ax = env.agent_pos
    coords = np.argwhere(env._sku_map > 0)
    if coords.size == 0:
        return env.goal_pos
    # choose by Manhattan distance
    dists = [abs(int(y) - ay) + abs(int(x) - ax) for (y, x) in coords]
    return tuple(coords[int(np.argmin(dists))])


def greedy_step(env: SariSandboxEnv):
    """One greedy action toward nearest target; pick if within radius."""
    ay, ax = env.agent_pos
    ty, tx = nearest_target(env)
    # if within pick radius, pick
    if abs(ay - ty) + abs(ax - tx) <= env.cfg.pick_radius:
        return 5  # pick
    # move to reduce Manhattan distance (prefer vertical first)
    moves = []
    if ty < ay:
        moves.append(1)  # up
    if ty > ay:
        moves.append(2)  # down
    if tx < ax:
        moves.append(3)  # left
    if tx > ax:
        moves.append(4)  # right
    # try moves in order, skip into shelves
    for a in moves:
        dy, dx = {1: (-1,0), 2: (1,0), 3: (0,-1), 4: (0,1)}[a]
        ny, nx = ay + dy, ax + dx
        if 0 <= ny < env.cfg.grid_size and 0 <= nx < env.cfg.grid_size and env._grid_layout[ny, nx] != 1:
            return a
    # fallback: noop
    return 0


def run_debug_episode(grid_size=9, n_shelves=10, n_skus=3, max_steps=10, seed=0):
#    env = SariSandboxEnv(TaskConfig(grid_size=grid_size, n_shelves=n_shelves, n_skus=n_skus, max_steps=max_steps), seed=seed)
    env = make_env(seed=seed)
    obs, _ = env.reset()
    print("t=-1", env.render(), sep="")
    for t in range(max_steps):
        a = greedy_step(env)
        print(a)
        obs, r, term, trunc, info = env.step(a)
        inv = info["inventory"].tolist()
        y, x = info["pos"]
        picked = info["picked_idx"]
        picked_str = f" picked=SKU{picked+1}" if picked >= 0 else ""
        print(f"t={t:03d} act={ACTION_NAMES[a]:>5} r={r:+.3f} pos=({y},{x}) inv={inv}{picked_str}")
        print(env.render())
        if RENDER_DELAY > 0:
            time.sleep(RENDER_DELAY)
        if term or trunc:
            break
    print("done.")


if __name__ == "__main__":
    run_debug_episode(max_steps=3, seed=42)
