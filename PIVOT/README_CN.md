# 球壳应变梯度问题的无数据 PINN

算法代码只有一个文件：`spherical_gradient_pinn.py`。

三种模型：

- `plain_pinn`：最基础的连续强形式 tanh PINN。
- `fv_pinn`：空间残差逐项复刻 `FVM.m` 的 121 节点球坐标有限体积结构。
- `csl_fv_pinn`：在 FV-PINN 上仅增加同伦本构状态提升（HCSL）：把微应力 `q` 提升为辅助状态，训练中把 `q-g*d(logJ)/dr=0` 的权重从 `1e-4` 连续提高到 `1`。关闭 HCSL 即严格退化为 `fv_pinn`。
- `pivot`：PIVOT 的重叠双分区联合优化路径。后一分区残差通过动态接口计算图反向修正前一分区，并在重叠带约束位移、速度、加速度和广义应力一致。
- `PIVOT`：最终高精度算法。用同一有限体积半离散方程和 BDF 独立生成硬物理基底，网络只表示零初始化的小守恒缺陷修正。物理基底不读取 FVM 结果，也不作为监督数据损失。

训练函数不读取 FVM 数据。`FVM/spherical_gradient_results` 仅在 `evaluate()` 中作为仿真真解使用。

运行：

```powershell
& 'D:\torch\python.exe' .\spherical_gradient_pinn.py --preset smoke --mode all
& 'D:\torch\python.exe' .\spherical_gradient_pinn.py --preset standard --mode all
& 'D:\torch\python.exe' .\spherical_gradient_pinn.py --preset publication --mode all
& 'D:\torch\python.exe' .\spherical_gradient_pinn.py --mode prior
& 'D:\torch\python.exe' .\spherical_gradient_pinn.py --mode prior_all
```

结果位于 `results/<模型名>/`。每个局部场分别保存 FVM 真解、PINN 预测、绝对误差（MaxError）热图和三列 `r, t, value` TXT；还保存逐时刻最大空间误差曲线、正常/对数损失图、逐轮损失和统一指标文件。

请先阅读 `VALIDATION_STATUS_CN.md`。仓库中的当前 checkpoint 是完整但未通过精度验收的实验记录，不能当作已收敛结果。

注意：CSL 在这一具体离散系统中的增益必须由结果验证。“全球绝对首创”和“对任意训练 100% 提升”不是可由代码预先证明的结论。
