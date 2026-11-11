# SES：等间隔保留整期

在当前论文方法上只改一件事：32 个连续睡眠期里，哪些整期被采集不再每夜重抽，而改成与 1 秒片段相同的等间隔公式。部署相位固定为 0。每个 32 期窗口采集的秒数是常数，整晚电量在记录前就能算出。

这不是「32 期全采集、只在期内减采样」。全信号（100%）才保留全部 32 期。72%、40%、10% 的可见期数、可见秒数与 MASS Table 1 相同，改变的只是整期的位置。

下标公式（期内、期外共用）：

```text
idx[i] = floor((phase + i * N / K) mod N),  i = 0 .. K-1
```

部署时 `phase = 0`。训练时相位只在一个步长内平移，`步长种数 = max(1, N // K)`；步长不足 2 时，训练和部署是同一张表。期内仍以概率 0.25 在训练批次中混入随机片段，测试时关闭。整期从不随机抽取。

## 部署时刻表

窗口与记录起点对齐，长 32 期（16 分钟），不重叠。绝对期号 `t` 采集的条件是 `(t mod 32)` 落在下表的保留整期里。整晚 `T` 期的采集秒数为

```text
|{t : (t mod 32) 属于保留整期}| × 期内保留秒数
```

`T` 是 32 的倍数时，就是 `(T / 32) × 下表最后一列`。这个数的方差为 0。

| 名义信号量 | 保留整期 | 待机整期 | 期内保留的秒 | 每 32 期采集秒数 | 实际比例 |
|---|---|---|---|---|---|
| 100% | 0–31 全部 | 无 | 0–29 全部 | 960 | 1.00 |
| 72% | 除 7, 15, 23, 31 外的 28 期 | 7, 15, 23, 31 | 除 4, 9, 14, 19, 24, 29 外的 24 秒 | 672 | 0.70 |
| 40% | 25 期，见下方列表 | 4, 9, 13, 18, 22, 27, 31 | 0, 2, 4, …, 28 | 375 | 0.390625 |
| 10% | 0, 2, 4, …, 30 | 1, 3, 5, …, 31 | 0, 5, 10, 15, 20, 25 | 96 | 0.10 |

40% 的保留整期为：

```text
0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30
```

72% 和 40% 的整期步长小于 2，整数相位只有 0，训练不会改这张表。10% 的整期在训练时于偶数期 / 奇数期两种表之间平移，部署固定为偶数期。40% 和 10% 的期内秒相位仍按原方法在训练时平移，部署固定为上表。

打印与代码核对过的时刻表：

```text
python -m SES --schedule
```

## 和已有结果的关系

- `--pi random_inter` 是期内等间隔、时期间随机的对照，可见数量与上表相同。
- 32 期全部保留、只减少 1 秒片段的几何与这里的 72% / 40% / 10% 不同，不能放在同一列。
- 命令里的 `--budget 0.72` 是名义 72%。实现上是 24/30 × 28/32 = 0.70。不要把 0.70 传给 `--budget`。

训练配方为 Lion、余弦 warmup、100 epoch、损失 = CE + 2×(1−cosine) + 0.5×transition CE、按准确率选 checkpoint、seed 2025。划分在 `data.py`：DREAMS 20 人留一，其余 20 折。batch size 为 8 / 16 / 64 / 128。

测试集尾部不足 32 期时仍沿用 MASS 的借用拼接，只影响最后一个窗口。

## 运行

在仓库根目录下执行。先确认时刻表和前向：

```text
python -m SES --schedule
python -m SES --smoke
```

论文主实验是 `--pi equal`（默认）。一组实验 = 一个数据集 × 一个信号量 × 20 折。下面这一条会按截图中的顺序串行跑完全部 16 组；某一组失败时继续后面的组，进度写在 `SES/results/_queue_<时间>/queue.log`。

```text
python -m SES --all
```

DREAMS：

```text
python -m SES --dataset dreams --budget 0.72
python -m SES --dataset dreams --budget 0.40
python -m SES --dataset dreams --budget 0.10
python -m SES --dataset dreams --budget 1.0
```

Sleep-EDF-20：

```text
python -m SES --dataset edf20 --budget 0.72
python -m SES --dataset edf20 --budget 0.40
python -m SES --dataset edf20 --budget 0.10
python -m SES --dataset edf20 --budget 1.0
```

Sleep-EDF-78：

```text
python -m SES --dataset edf78 --budget 0.72
python -m SES --dataset edf78 --budget 0.40
python -m SES --dataset edf78 --budget 0.10
python -m SES --dataset edf78 --budget 1.0
```

SHHS：

```text
python -m SES --dataset shhs --budget 0.72
python -m SES --dataset shhs --budget 0.40
python -m SES --dataset shhs --budget 0.10
python -m SES --dataset shhs --budget 1.0
```

100% 四条是全信号参照，论文若仍不报告这一列，可以不跑。

单折连通检查（2 个 epoch，不能当结果）：

```text
python -m SES --dataset dreams --budget 0.40 --fast --fold 0
```

表 3 消融固定名义 40%。四个变体的时期间整期都是相位 0，待机整期为 4、9、13、18、22、27、31。`--pi random` 是 SES-Random，训练和测试都重抽 1 秒片段。`--pi head` 是 SES-Head，每个可见整期只保留前 15 秒。`--pi uniform` 是 SES-Uniform，时期内用相位 0 的等间隔格子，训练不平移相位，也不混入随机片段。`--pi equal` 是 SES-MASS。这三组消融不使用 `--mix_random`。

```text
python -m SES --dataset dreams --pi random --budget 0.40
python -m SES --dataset dreams --pi head --budget 0.40
python -m SES --dataset dreams --pi uniform --budget 0.40
python -m SES --dataset dreams --pi equal --budget 0.40
```

Sleep-EDF-20、Sleep-EDF-78、SHHS 把 `--dataset` 换成 `edf20`、`edf78`、`shhs`。一次排队跑完 4 个数据集 × 4 个变体：

```text
python -m SES --ablation
```

可选对照，时期内等间隔、时期间随机：

```text
python -m SES --dataset dreams --pi random_inter --budget 0.40
```

结果写在 `SES/results/<dataset>_<pi>_B0.40_<时间>/`。主实验的 `--pi` 默认为 `equal`。其中 `schedule.json` 是该次运行的时刻表，`summary.json` 给出 ACC / MF1 / κ / MGm。`obs_ratio` 应为上表的实际比例，`obs_ratio_std` 应为 0。
