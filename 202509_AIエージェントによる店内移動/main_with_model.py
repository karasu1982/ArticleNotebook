from stable_baselines3 import PPO
from sari_sandbox import SariSandboxEnv, TaskConfig, ACTION_NAMES

# 環境は学習時と“同じ設定”に合わせるのが基本
env = SariSandboxEnv(TaskConfig(grid_size=9, n_shelves=10, n_skus=3, max_steps=200), seed=42)

model_path = "./models/sari_ppo_20250926_195719.zip"  # あなたの保存名に置換
model = PPO.load(model_path)

obs, _ = env.reset()
for t in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(int(action))
    if info.get("picked_idx", -1) >= 0:
        print(f"[t={t}] ピック成功 → SKU{info['picked_idx']+1}, inv={info['inventory'].tolist()}, r={r:+.3f}")
    if term or trunc:
        break

print(env.render())  # 最終盤面をテキストで確認
