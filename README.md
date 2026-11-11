# SES-MASS

结构化等间隔采样的掩码感知睡眠分期（Structured Equal-interval Sampling Mask-Aware Sleep Staging）。

在 MASS 的掩码感知分期网络上，把 30 秒睡眠期内的 1 秒片段和睡眠期之间的整期都改成等间隔选取。部署相位固定为 0，采集与待机时刻可以在记录前排定。

## 代码

| 路径 | 内容 |
|---|---|
| `SES/` | 等间隔采样、训练与消融实验 |
| `structure/` | MASS 分期网络：全局提示、时期内编码、时期间编码 |
| `utils.py` | 读取预处理后的 `.mat` 并组成 batch |
| `lion_pytorch/` | 训练使用的 Lion 优化器 |

`structure/` 保留 MASS 实现的 MIT 许可，版权声明见仓库根目录 `LICENSE`。Lion 优化器的许可见 `lion_pytorch/LICENSE`。

## 运行

在仓库根目录执行：

```text
pip install -r requirements.txt
python -m SES --schedule
python -m SES --smoke
python -m SES --dataset dreams --budget 0.40
```

采样规则、四档信号量和消融命令见 [SES/README.md](SES/README.md)。

预处理后的脑电不在本仓库中。程序读取 `dataset/DREAMS`、`dataset/EDF20`、`dataset/EDF78`、`dataset/SHHS` 下的 `.mat` 文件。
