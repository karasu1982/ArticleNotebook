# ==============================================
# File: debug_episode.py
# Step-by-step run with a better greedy planner (no needless noop)
# ==============================================

import time
import collections
import numpy as np
from sari_sandbox import SariSandboxEnv, TaskConfig, ACTION_NAMES, make_env

RENDER_DELAY = 0.0  # 0.1~0.3 にするとコマ送り
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # up,down,left,right

def needed_sku_coords(env: SariSandboxEnv):
    """まだ必要なSKUだけを抽出して座標を返す"""
    coords = []
    ys, xs = np.where(env._sku_map > 0)
    for y, x in zip(ys, xs):
        sku = int(env._sku_map[y, x])
        if env.inventory[sku - 1] < env.picklist[sku - 1]:
            coords.append((int(y), int(x)))
    return coords

def nearest_target(env: SariSandboxEnv):
    """
    残SKUがあれば最も近い棚（マンハッタン距離）。
    無ければゴールへ。
    """
    ay, ax = env.agent_pos
    coords = needed_sku_coords(env)
    if not coords:
        return env.goal_pos
    dists = [abs(y - ay) + abs(x - ax) for (y, x) in coords]
    return coords[int(np.argmin(dists))]

def in_bounds(env: SariSandboxEnv, y: int, x: int) -> bool:
    return 0 <= y < env.cfg.grid_size and 0 <= x < env.cfg.grid_size

def passable(env: SariSandboxEnv, y: int, x: int) -> bool:
    # 棚(1)は不可侵
    return env._grid_layout[y, x] != 1

def bfs_next_move(env: SariSandboxEnv, target):
    """
    最短経路の最初の一歩をBFSで求める。
    到達不能なら None。
    """
    start = env.agent_pos
    if start == target:
        return None
    q = collections.deque([start])
    came = {start: None}
    while q:
        y, x = q.popleft()
        for a, (dy, dx) in MOVES.items():
            ny, nx = y + dy, x + dx
            if not in_bounds(env, ny, nx) or not passable(env, ny, nx):
                continue
            if (ny, nx) in came:
                continue
            came[(ny, nx)] = (y, x, a)
            if (ny, nx) == target:
                # 復元して最初の一手を返す
                cur = (ny, nx)
                while came[cur] and came[came[cur][0:2]]:
                    cur = came[cur][0:2]
                # cur は start の隣マスになっているはず
                first = came[(ny, nx)]
                # 直近に遡って「startからの最初のアクション」を求める
                # 1ステップ目を直接返すため、親が start のところを見つける
                prev = (ny, nx)
                while came[prev] and came[prev][0:2] != start:
                    prev = came[prev][0:2]
                return came[prev][2]  # action id
            q.append((ny, nx))
    return None

def greedy_step(env: SariSandboxEnv):
    """
    近傍ターゲットへ貪欲に進む。隣接なら pick。
    縦→横で塞がれていたら 横→縦も試し、それでもダメなら BFS で回避。
    """
    ay, ax = env.agent_pos
    ty, tx = nearest_target(env)

    # 隣接以内なら pick で試す
    if abs(ay - ty) + abs(ax - tx) <= env.cfg.pick_radius:
        return 5  # pick

    # 縦→横の順で貪欲
    plan_orders = []
    moves_main = []
    if ty < ay: moves_main.append(1)  # up
    if ty > ay: moves_main.append(2)  # down
    if tx < ax: moves_main.append(3)  # left
    if tx > ax: moves_main.append(4)  # right
    plan_orders.append(moves_main)

    # 横→縦の順（代替順）
    moves_alt = []
    if tx < ax: moves_alt.append(3)
    if tx > ax: moves_alt.append(4)
    if ty < ay: moves_alt.append(1)
    if ty > ay: moves_alt.append(2)
    plan_orders.append(moves_alt)

    # 両順序で通れる手を探す
    for moves in plan_orders:
        for a in moves:
            dy, dx = MOVES[a]
            ny, nx = ay + dy, ax + dx
            if in_bounds(env, ny, nx) and passable(env, ny, nx):
                return a

    # それでも詰んでいれば BFS で最短回避
    a = bfs_next_move(env, (ty, tx))
    if a is not None:
        return a

    # 本当に何もできない時だけ noop
    return 0

def run_debug_episode(grid_size=9, n_shelves=10, n_skus=3, max_steps=100, seed=42):
    # make_env() は環境を返す実装にしてある想定
    env = make_env(grid_size=grid_size, n_shelves=n_shelves, n_skus=n_skus, max_steps=max_steps, seed=seed)
    obs, _ = env.reset()
    print("t=-1\n" + env.render())
    for t in range(max_steps):
        a = greedy_step(env)
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
    run_debug_episode()
