# 命名实体识别（NER）任务

本项目基于 PyTorch 与 HuggingFace Transformers 实现中文命名实体识别（NER）任务，在 Weibo、MSRA 两个中文 NER 数据集上，对比`bert-base-chinese`与`chinese-bert-wwm`两种预训练权重的模型效果。使用 SwanLab 进行实验指标记录与可视化，提供训练脚本、推理预测脚本，可直接复现实验结果。

## 项目结构

```
nlp-demo2
├── src
│   ├── model.py        # BertNERModel模型定义，BERT+Dropout+分类头
│   ├── dataset.py      # 数据集加载、BIO标签转换、token对齐、collate_fn
│   ├── train.py        # 训练主脚本，早停策略、AdamW优化器
│   ├── utils.py        # 评价指标工具类，BIO实体解码、F1/Precision/Recall计算
│   ├── config.py       # 配置管理模块
│   └── predict.py      # 推理脚本，输入句子抽取命名实体
├── checkpoints         # 保存各实验最优模型权重与实验配置json
├── .gitignore          # 忽略大权重、日志、缓存文件
└── README.md           # 项目说明文档
```

## 环境依赖

```
torch>=2.0
transformers
swanlab
```

安装命令：

```
pip install torch transformers swanlab
```

## 实验说明

### 实验设置

- 预训练模型：`bert-base-chinese`、`hfl/chinese-bert-wwm`
- 数据集：Weibo 、MSRA
- 超参统一：`batch_size=8`，`lr=2e-5`，`epoch=10`，`max_len=128`，dropout=0.2
- 优化器：AdamW
- 早停策略：基于验证集 F1 指标，防止过拟合
- 评价指标：Precision、Recall、F1

### 实验结果汇总
#### 训练可视化（SwanLab）
本实验使用SwanLab记录训练过程，可查看训练损失、测试集F1、精确率、召回率随epoch变化曲线。
- 训练loss曲线：观察模型收敛与过拟合情况
- 测试指标曲线：观察F1、Precision、Recall变化趋势

#### weibo_bert_base_v3
<img width="1094" height="681" alt="image" src="https://github.com/user-attachments/assets/84ca13d3-7a72-4e06-bc41-c81603fcaa12" />

**训练现象**

- 验证损失dev/loss：第2轮降到最低，之后持续走高，上升幅度比MSRA两组更大，过拟合现象更明显。微博数据集样本噪声大、样本数量少，更容易过拟合。
- 指标：Precision震荡剧烈；Recall稳步上升；F1在前6轮持续上升，之后小幅回落。
- 训练过程：微博文本口语化、网络用语多、实体边界模糊，识别难度显著高于MSRA。整体F1远低于MSRA实验。
- 结论：BERT-base在微博数据集上可以学到实体特征，但受噪声影响，验证集指标波动大。

#### weibo_bert_wwm_v3
<img width="1005" height="602" alt="image" src="https://github.com/user-attachments/assets/73df6322-2d49-41ef-a6f4-5e84bb472116" />

**训练现象**

- 验证损失dev/loss：第2轮降到最低点，后续持续上升，过拟合趋势与weibo_bert_base接近。
- 指标：Precision存在明显震荡；Recall稳定上升；F1前期快速上升，在第5轮达到峰值后小幅波动下降。
- 训练过程：WWM模型在微博数据集上和base模型性能差距很小，最终F1略低于weibo_bert_base。全词掩码预训练没有带来提升。
- 结论：针对微博这类碎片化社交媒体文本，WWM预训练策略没有优势。
#### msra_bert_base_v3
<img width="1038" height="645" alt="image" src="https://github.com/user-attachments/assets/2d5e67bc-c5c6-41da-b5c4-20e3d22505f4" />

**训练现象**

- 验证损失 dev/loss：第1个step快速下降，第2步达到最低点，之后持续上升。说明模型在2轮之后开始在验证集上出现过拟合，验证损失逐步走高。
- 指标（Precision/Recall/F1）：前期快速上涨，中间有明显震荡波动，F1在第5轮达到峰值，之后小幅回落。
- 训练过程：模型在前5个 epoch 快速学习 MSRA 规范文本的实体特征，精确率波动比较明显；召回率整体稳步走高，稳定性更好。
- 结论：该组取得四组里最高的F1，虽然验证loss后期上升，但综合指标最优，在MSRA数据集上BERT-base效果优于WWM。

#### msra_bert_wwm_v3
<img width="1055" height="614" alt="image" src="https://github.com/user-attachments/assets/7f4ee8ef-ea3b-4b5f-b125-11cc2d24bc30" />

**训练现象**

- 验证损失dev/loss：首轮快速下降，第2轮到达最低点，后续震荡上升，同样出现过拟合趋势。
- 指标：Precision震荡幅度很大，高低起伏明显；Recall持续稳步上升；F1在第5轮达到最高点，之后下降。
- 训练过程：WWM的全词掩码预训练方式，在MSRA规范新闻文本上没有太多优势。指标波动比base版本更大，最终F略低于`msra_bert_base_v3`。
- 结论：MSRA数据集下，`bert-base`优于`bert-wwm`。

| 实验名称 | 预训练模型 | 数据集 | Precision | Recall | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| weibo_bert_base_v3 | bert-base-chinese | Weibo | 0.6610 | 0.6691 | 0.6650 |
| weibo_bert_wwm_v3 | chinese-bert-wwm | Weibo | 0.6570 | 0.6666 | 0.6618 |
| msra_bert_base_v3 | bert-base-chinese | MSRA | 0.9154 | 0.9103 | 0.9129 |
| msra_bert_wwm_v3 | chinese-bert-wwm | MSRA | 0.9123 | 0.9080 | 0.9101 |
> 
> 结果分析：
> 
> 
> 1. MSRA 数据集上模型整体效果远好于 Weibo 数据集。Weibo 数据集为社交媒体文本，口语化、实体类型细粒度划分（区分 NAM/NOM），识别难度更高，F1 明显更低。
> 2. MSRA 数据集：`bert-base-chinese` 略优于 wwm 版本；
> 3. Weibo 数据集：wwm 与 base 模型性能差距很小，wwm 略微低于 base。


## 训练命令

```
# Weibo + bert-base
PYTHONPATH=. python src/train.py --exp_name weibo_bert_base_v3 --dataset weibo --pretrain_name bert-base-chinese --epoch 10 --batch_size 8 --seed 42 --lr 2e-5 --dropout 0.2

# Weibo + chinese-bert-wwm
PYTHONPATH=. python src/train.py --exp_name weibo_bert_wwm_v3 --dataset weibo --pretrain_name hfl/chinese-bert-wwm --epoch 10 --batch_size 8 --seed 42 --lr 2e-5 --dropout 0.2

# MSRA + bert-base
PYTHONPATH=. python src/train.py --exp_name msra_bert_base_v3 --dataset msra --pretrain_name bert-base-chinese --epoch 10 --batch_size 8 --seed 42 --lr 2e-5 --dropout 0.2

# MSRA + chinese-bert-wwm
PYTHONPATH=. python src/train.py --exp_name msra_bert_wwm_v3 --dataset msra --pretrain_name hfl/chinese-bert-wwm --epoch 10 --batch_size 8 --seed 42 --lr 2e-5 --dropout 0.2
```

## 推理预测

修改`predict.py`中配置文件路径，直接运行脚本，输入文本自动抽取实体：

```
python src/predict.py
```

示例输出：

```
输入句子: 张三在北京上学，李四去上海出差。
  [PER.NAM] 张三  (位置 0-1)
  [GPE.NAM] 北京  (位置 3-4)
  [PER.NAM] 李四  (位置 8-9)
  [GPE.NAM] 上海  (位置 11-12)
```
