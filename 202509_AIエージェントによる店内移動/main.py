from sari_sandbox import make_env
env = make_env(seed=42)  # pick_radius=1 が既定に
obs, _ = env.reset()
print(env.render())     # 数字(1/2/3)の隣に寄って

# 例: 100ステップ以内に、毎ステップ pick も試してみる
for t in range(100):
    # たとえばランダム移動＋たまにpick
    a = env.action_space.sample()
    if t % 3 == 0:  # こまめに pick してみる
        a = 5
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        break

print(env.render())     # inv=[...] が 1つ以上に増えていればOK
