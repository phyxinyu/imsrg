1.固定 quantum 太粗，后期小贡献 round 不出来

现在 quantum 是初始化时固定的：

quantum = L1(|H1| + |H2|) / N_initial
N=1e7 虽然大，但后期 IMSRG 接近收敛时：

ds * contribution / quantum
很多 source contribution 的期望 walker 数会远小于 1，只能靠 stochastic rounding 偶尔产生 ±1。这时候更新变成稀疏随机噪声，而不是平滑地继续下降。




2.当前没有 population control / quantum refinement

FCIQMC 里通常有 shift、population control、initiator 等机制控制 population 和噪声。现在这版是固定 quantum：

不 rescale
不调整 quantum
不做 population control
不做 deterministic correction
所以它更像第一版 proof-of-concept。到了后期，固定精度可能不够。