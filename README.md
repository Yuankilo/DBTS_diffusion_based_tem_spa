# Reviewer Package

这个目录只保留作者自有方法的最小可运行代码，不包含 baseline，也不包含任何私有数据。

## 保留内容

- 单一运行入口：`run_reviewer_forecast.py`
- 对应配置：`Config/reviewer_forecast.yaml`
- 运行该方法所需的核心模型与训练代码：`Models/`、`engine/`、`Utils/`
- 许可证与最小依赖：`LICENSE`、`requirements.txt`

## 私有数据放置位置

请将私有数据放入 `private_data/` 目录，并按下面方式提供：

- `series.csv`
  必需。主时间序列数据，形状应与当前模型一致。
- `adjacency.csv`
  必需。邻接矩阵。
- `external_features.csv`
  可选。时间步外部特征；如果不提供，代码会自动使用全零特征。
- `adj_new.csv`
  可选。节点侧辅助矩阵；如果不提供，代码会自动使用全零矩阵。

仓库中不会包含这些文件本体，只保留放置说明。

## 当前代码的结构约束

为保持与你现有实现一致，当前版本默认按以下维度运行：

- 序列长度：`48`
- 预测长度：`24`
- 节点/变量数：`282`
- 外部特征维度：`195`

这些维度同时受到模型代码中的固定层设置约束。如果审稿人要换成其他维度，需要进一步重构模型文件，而不仅仅是改配置。

## 运行方式

先安装依赖：

```bash
pip install -r requirements.txt
```

示例运行：

```bash
python run_reviewer_forecast.py \
  --series-csv private_data/series.csv \
  --adj-csv private_data/adjacency.csv \
  --external-csv private_data/external_features.csv \
  --adj-new-csv private_data/adj_new.csv \
  --output-dir review_outputs/reviewer_run
```

如果没有外部特征或辅助矩阵，也可以只传必需项：

```bash
python run_reviewer_forecast.py \
  --series-csv private_data/series.csv \
  --adj-csv private_data/adjacency.csv
```

## 输出

结果默认写入你指定的 `--output-dir`，包括：

- `metrics.csv`
- `predictions.npy`
- `ground_truth.npy`
- `forecast_mask.npy`

## 说明

- 这个包已经移除了 baseline、测试草稿脚本、示例数据和各类本地产物。
- 现在审稿人只需要看一个脚本入口即可。
