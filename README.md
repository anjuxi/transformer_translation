# Transformer-中英文翻译

这个项目实现了一个完整的Transformer架构神经网络，采用自注意力机制（Self-Attention）进行端到端的机器翻译。模型包含编码器-解码器结构、多头注意力机制、位置编码等核心组件，适用于短句级别的中英文翻译任务。

## 项目结构

```
.
├── README.md                          # 项目说明文档
├── transformer-translation.ipynb      # Jupyter Notebook版本（推荐）
└── data.csv                           # 训练数据集（中英文平行语料）
```

## 数据集

训练数据为CSV格式，包含以下列：

- 英文句子
- 中文翻译
- 许可证信息

数据来源于`https://www.kaggle.com/datasets/thespringcomes/translationcmn3`项目，包含日常用语短句的平行语料。

## 模型架构

| 组件         | 配置 |
| ------------ | ---- |
| 词嵌入维度   | 256  |
| 注意力头数   | 4    |
| 编码器层数   | 2    |
| 解码器层数   | 2    |
| 前馈网络维度 | 512  |
| Dropout比率  | 0.2  |

模型参数量约2-3M，适合在单张GPU（如T4）上快速训练。

## 快速开始

1. 打开 `transformer-translation.ipynb`
2. 按顺序运行各个Cell
3. 查看训练过程和翻译效果

## 训练流程

1. **数据加载与预处理**：清洗文本、构建词汇表
2. **模型初始化**：创建Transformer模型
3. **模型训练**：支持多GPU训练、断点续训
4. **效果测试**：翻译测试样例、绘制损失曲线

### 断点续训

将 `CONTINUE_TRAIN` 设置为 `True`，模型会从已保存的checkpoint继续训练：

```python
CONTINUE_TRAIN = True;
```

## 测试样例

模型训练完成后，可测试以下英文句子的翻译效果：

- Hello!
- How are you?
- Thank you very much.
- I love you.
- Good morning!

## 硬件要求

- **最低配置**：CPU即可运行
- **推荐配置**：NVIDIA GPU（支持CUDA）
- **最佳配置**：双T4 GPU（自动启用DataParallel多卡训练）

## 输出文件

- `transformer_model.pt`：保存的最佳模型
- `training_curve.png`：训练损失曲线图

## 技术细节

### 数据处理

- 英文文本：小写转换、特殊字符清洗
- 中文文本：字符级分词（字与字之间添加空格）
- 词汇表：基于词频过滤（最小词频=2）

### 训练策略

- 优化器：Adam（lr=0.0001, betas=(0.9, 0.98)）
- 学习率调度：ReduceLROnPlateau
- 梯度裁剪：max_norm=1.0
- 损失函数：CrossEntropyLoss（忽略padding）

### 推理方式

- 贪婪解码（Greedy Decoding）
- 支持批量翻译

## 许可证

本项目采用MIT许可证开源。

## 联系方式

**谙弆悕博士（Ailan Anjuxi）**

- [邮箱](@anjuxi.ME@outlook.com)：anjuxi.ME@outlook.com
- [SIP电话](@sip:anjuxi@sip.linphone.org)：sip:anjuxi@sip.linphone.org
