# CisTempoForge

CisTempoForge 是一个用于时间序列 cis-regulatory 表达预测的可安装 Python 包。它把原先
D0–D5 研究脚本中的数据集、模型、SIMPLE_HUBER loss、训练、选模、预测和 checkpoint
接口拆成可复用模块，适合从 Notebook 或独立训练工程调用。

## 安装

```bash
python -m pip install -e '.[test]'
```

正式 CUDA 环境需要自行安装与驱动匹配的 PyTorch。包支持 Python 3.10 及以上版本。

## 快速训练

```python
from cistempoforge import fit, load_config, load_model

config = load_config("config.json")
config.training.learning_rate = 1e-4
config.training.epochs = 40

result = fit(config, output_dir="runs/d0_d5_seed42")
model, checkpoint = load_model(result.best_checkpoint, device="cuda")
```

`fit` 只构造配置指定的 train 和 validation dataset；test split 不参与训练、early
stopping 或 checkpoint selection。测试集预测应在选模结束后由调用方显式创建 dataset，
再调用 `predict`。

已有 `best.pt`、`last.pt`、`history.csv` 或 `manifest.json` 时，新的训练默认拒绝覆盖。
中断后可使用：

```python
result = fit(config, "runs/d0_d5_seed42", resume_from="runs/d0_d5_seed42/last.pt")
```

完整示例见 `examples/train_from_python.py`、`examples/notebook_quickstart.ipynb` 和
`examples/d0_d5_config.json`。

## 输入数据契约

配置中的路径均为调用进程可解析的显式路径，不依赖仓库位置：

- `model_inputs.npz`：包含 `dna [peaks,4,length]`、`atac [peaks,days,length]`、
  `target [genes,days]`、`peak_mask [genes,max_peaks]`、同形状的
  `gene_peak_index`，以及 `atac_channel_mean`、`atac_channel_std`。
- `metadata.parquet`：每个 gene 一行，至少包含 `split` 和
  `n_selected_promoter_peaks`。
- distance `.npy`：形状为 `[genes,max_peaks]`。
- structure `.npz`：包含 `features` 和 `masks`，二者形状均为 `[genes,8]`。

target normalization 只由 train split 和选定的 `day_indices` 估计。`n_days` 必须与
`day_indices` 长度一致，可以使用 6、7 或其他不少于 2 的时间点。

## 稳定 API

高层入口为 `fit`、`load_config` 和 `load_model`。需要自定义训练循环时，可直接使用
`CisTempoForgeModel`、`CisTempoForgeDataset`、`load_data`、`make_loader`、
`simple_huber_loss`、`train_epoch`、`evaluate`、`predict`、`save_checkpoint` 和
`trajectory_metrics`。

checkpoint 格式从本包 `0.1.0` 开始版本化；历史 D0–D5 脚本产生的 checkpoint 不属于
新格式，也不承诺直接兼容。

## 训练产物

调用方拥有 `output_dir`。包写入：

- `best.pt`：按预注册规则选择的 checkpoint；
- `last.pt`：最近 epoch 的完整恢复状态；
- `.epoch_checkpoints/`：精确选模与中断恢复所需的逐 epoch 状态；
- `history.csv` 和 `manifest.json`。

选模规则为：先保留 centered RMSE 不超过全程最低值 `1 + tolerance` 的 epoch，再选择
median trajectory Pearson 最高者，最终以最早 epoch 打破平局。默认 tolerance 为 0.5%。

## License

MIT License. Copyright (c) 2026 yuanlizhanshi.
