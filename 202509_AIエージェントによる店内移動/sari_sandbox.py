# ==============================================
# File: sari_sandbox.py
# 小売店の棚を模したシンプルなグリッド環境
# エージェントが移動・商品ピック・ゴール到達を学習するためのGymnasium環境
#
#
# 概要：
# 小売店の棚レイアウトを簡略化した N×N グリッド。
# エージェントは上下左右に移動し、隣接する棚からSKUをピック（pick_radius=1）する。
# 全SKUを1個ずつ集めてから、右下のゴールに到達すれば成功（エピソード終了）。
# Gymnasium 形式なので、RLアルゴリズム（PPOなど）で学習できる。
# ==============================================

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 行動をわかりやすくするための名前リスト
ACTION_NAMES = ["noop", "up", "down", "left", "right", "pick"]

# -------------------------------
# 乱数シードの固定
# -------------------------------
def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

# タスク設定をまとめるデータクラス
@dataclass
class TaskConfig:
    grid_size: int = 9               # グリッドの大きさ（N×N）
    n_shelves: int = 10              # 棚の数
    n_skus: int = 3                  # SKU（商品）種類数
    max_steps: int = 200             # 1エピソードの最大ステップ数
    pick_radius: int = 1             # ピックできる距離（1なら隣接OK）
    # 報酬設計
    reward_step: float = -0.005      # 1ステップごとの負報酬（時間コスト）
    reward_noop: float = -0.01       # ←追加: noop（待機）ペナルティ
    reward_pick: float = 1.0         # 正しいピックに成功したときの報酬
    reward_goal: float = 2.0         # 全ピック完了後にゴール到達した報酬
    reward_wrong_pick: float = -0.1  # 間違ったピックをしたときの罰
    reward_wall: float = -0.02       # 壁や棚にぶつかったときの罰
    with_obstacles: bool = True      # 棚を配置するかどうか


class SariSandboxEnv(gym.Env):
    """
    小売店を模したグリッド環境。
    - エージェントは上下左右に移動し、隣接する棚の商品をピックできる。
    - すべてのSKUを集めてからゴールに到達すると成功。

    観測（Dict型）:
      grid: (H, W, 3) の擬似RGBマップ
        - ch0: レイアウト（0:空, 50:棚, 100:ゴール, 150:エージェント）
        - ch1: SKU ID (0:なし, 1..K)
        - ch2: ピック済みマスク
      inventory: (K,) 各SKUの所持数
      goal: (2,) ゴール位置を正規化した座標

    行動（Discrete 6）:
      0: noop（待機）, 1: 上, 2: 下, 3: 左, 4: 右, 5: ピック
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, task: TaskConfig | None = None, seed: Optional[int] = None):
        super().__init__()
        self.cfg = task or TaskConfig()
        seed_everything(seed)

        N = self.cfg.grid_size
        # 行動空間
        self.action_space = spaces.Discrete(6)

        # 観測空間
        self.obs_h = N
        self.obs_w = N
        self.grid_channels = 3
        grid_space = spaces.Box(low=0, high=255, shape=(self.obs_h, self.obs_w, self.grid_channels), dtype=np.uint8)
        inv_space = spaces.Box(low=0.0, high=1.0, shape=(self.cfg.n_skus,), dtype=np.float32)
        goal_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "grid": grid_space,
            "inventory": inv_space,
            "goal": goal_space,
        })

        # 内部状態
        self._grid_layout = None   # 棚の配置（0:空,1:棚）
        self._sku_map = None       # SKUの配置（0:なし,1..K）
        self._picked_mask = None   # ピック済みフラグ
        self.agent_pos = (0, 0)
        self.goal_pos = (N-1, N-1)
        self.picklist = np.ones(self.cfg.n_skus, dtype=np.int32)  # 各SKUを1個ずつ必要
        self.inventory = np.zeros(self.cfg.n_skus, dtype=np.int32)
        self.steps = 0
        self._last_picked_idx = -1  # 直前にピックしたSKU（デバッグ用）

    # -------------------------------
    # ワールド生成関連
    # -------------------------------
    def _empty_map(self):
        N = self.cfg.grid_size
        return np.zeros((N, N), dtype=np.uint8)

    def _place_obstacles_and_shelves(self):
        """棚をランダムに配置。開始(0,0)とゴール(N-1,N-1)は避ける"""
        N = self.cfg.grid_size
        grid = self._empty_map()
        placed = 0
        tries = 0
        while placed < self.cfg.n_shelves and tries < 10000:
            y = np.random.randint(1, N-1)
            x = np.random.randint(1, N-1)
            if (y, x) in [(0, 0), (N-1, N-1)]:
                tries += 1
                continue
            if grid[y, x] == 0:
                grid[y, x] = 1
                placed += 1
            tries += 1
        return grid

    def _place_skus(self, layout):
        """棚の上にSKUを配置（1..Kを巡回）"""
        N = self.cfg.grid_size
        sku_map = np.zeros((N, N), dtype=np.uint8)
        shelf_positions = np.argwhere(layout == 1)
        np.random.shuffle(shelf_positions)
        k = 0
        for (y, x) in shelf_positions:
            sku_map[y, x] = (k % self.cfg.n_skus) + 1
            k += 1
            if k >= self.cfg.n_shelves:
                break
        return sku_map

    def _reset_world(self):
        """環境をリセット"""
        N = self.cfg.grid_size
        self._grid_layout = self._place_obstacles_and_shelves() if self.cfg.with_obstacles else self._empty_map()
        self._sku_map = self._place_skus(self._grid_layout)
        self._picked_mask = np.zeros_like(self._sku_map, dtype=np.uint8)
        self.agent_pos = (0, 0)
        self.goal_pos = (N-1, N-1)
        self.steps = 0
        self.inventory[:] = 0
        self._last_picked_idx = -1

    # -------------------------------
    # 状態チェック
    # -------------------------------
    def _at_goal(self) -> bool:
        return self.agent_pos == self.goal_pos

    def _all_picked(self) -> bool:
        return np.all(self.inventory >= self.picklist)

    # -------------------------------
    # 行動処理
    # -------------------------------
    def _move(self, dy: int, dx: int) -> float:
        """移動処理。壁や棚ならペナルティ"""
        N = self.cfg.grid_size
        y, x = self.agent_pos
        ny, nx = y + dy, x + dx
        if not (0 <= ny < N and 0 <= nx < N):
            return self.cfg.reward_wall
        if self._grid_layout[ny, nx] == 1:
            return self.cfg.reward_wall
        self.agent_pos = (ny, nx)
        return 0.0

    def _pick(self) -> float:
        """隣接する棚の商品をピック（マンハッタン距離<=pick_radius を探索）"""
        self._last_picked_idx = -1
        y, x = self.agent_pos
        r = int(self.cfg.pick_radius)
        N = self.cfg.grid_size
        target = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dy) + abs(dx) > r:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < N and 0 <= nx < N):
                    continue
                sku = int(self._sku_map[ny, nx])
                if sku == 0:
                    continue
                idx = sku - 1
                if self.inventory[idx] < self.picklist[idx]:
                    target = (ny, nx, idx)
                    break
            if target:
                break
        if target is None:
            return self.cfg.reward_wrong_pick
        ny, nx, idx = target
        self.inventory[idx] += 1
        self._picked_mask[ny, nx] = 1
        self._sku_map[ny, nx] = 0
        self._last_picked_idx = idx
        return self.cfg.reward_pick

    # -------------------------------
    # Gym API
    # -------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            seed_everything(seed)
        self._reset_world()
        obs = self._make_obs()
        return obs, {}

    def step(self, action: int):
        self.steps += 1
        reward = self.cfg.reward_step
        terminated = False
        truncated = False

        if action == 1:
            reward += self._move(-1, 0)
        elif action == 2:
            reward += self._move(1, 0)
        elif action == 3:
            reward += self._move(0, -1)
        elif action == 4:
            reward += self._move(0, 1)
        elif action == 5:
            reward += self._pick()
        else:  # action == 0 (noop)
            reward += self.cfg.reward_noop  # ← 追加：待機にコストを課す

        if self._all_picked() and self._at_goal():
            reward += self.cfg.reward_goal
            terminated = True

        if self.steps >= self.cfg.max_steps:
            truncated = True

        obs = self._make_obs()
        info = {
            "steps": self.steps,
            "action": int(action),
            "pos": tuple(self.agent_pos),
            "inventory": self.inventory.copy(),
            "picked_idx": int(self._last_picked_idx),
        }
        return obs, float(reward), terminated, truncated, info

    # -------------------------------
    # 観測・表示
    # -------------------------------
    def _make_obs(self) -> Dict[str, np.ndarray]:
        N = self.cfg.grid_size
        grid_img = np.zeros((N, N, self.grid_channels), dtype=np.uint8)
        # ch0: レイアウト＆エンティティ
        grid_img[..., 0] = 0
        grid_img[self._grid_layout == 1, 0] = 50
        gy, gx = self.goal_pos
        grid_img[gy, gx, 0] = 100
        ay, ax = self.agent_pos
        grid_img[ay, ax, 0] = 150
        # ch1: SKU ID
        grid_img[..., 1] = self._sku_map
        # ch2: ピック済みマスク
        grid_img[..., 2] = self._picked_mask

        inv = self.inventory.astype(np.float32)
        goal_vec = np.array([self.goal_pos[0] / N, self.goal_pos[1] / N], dtype=np.float32)
        return {"grid": grid_img, "inventory": inv, "goal": goal_vec}

    def render(self):
        N = self.cfg.grid_size
        out_lines: List[str] = []
        for y in range(N):
            row = []
            for x in range(N):
                if (y, x) == self.agent_pos:
                    row.append("A")
                elif (y, x) == self.goal_pos:
                    row.append("G")
                elif self._grid_layout[y, x] == 1:
                    if self._sku_map[y, x] > 0:
                        row.append(str(int(self._sku_map[y, x])))
                    else:
                        row.append("#")
                else:
                    row.append(".")
            out_lines.append(" ".join(row))
        inv = ",".join([str(int(v)) for v in self.inventory])
        out_lines.append(f"inv=[{inv}] steps={self.steps}")
        return "\n".join(out_lines)

# 環境生成関数（SB3で使える形式）
def make_env(task_type: str = "pick", seed: Optional[int] = None, **overrides):
    cfg = TaskConfig(**overrides)
    def _thunk():
        return SariSandboxEnv(task=cfg, seed=seed)
    return _thunk()