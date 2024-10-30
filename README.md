#### PJ介绍
本次pj是 基于预训练transformer模型的wnli分类，我最终通过更换模型为microsoft/deberta-v3-large,使用warmup和cosine策略，以及调整lr，bs，epoch等超参，实现了test的score达到85.6的效果
具体内容见pdf

#### 文件结构

```plaintext
project_name/
├── README.md                 # 项目说明文件，包含项目简介、安装方法、使用说明等
├── NLP practice pj ...pdf    # 主要说明文档
├── test.py                   # 用于测试模型，获取测试结果
├── train_data_argument.py    # 数据增强的训练代码
└── train.py                  # 训练模型的主脚本
```

#### 运行命令

```python
python -u train_old.py --bs=16  --output=./output/v1 --epoch=5 --model=microsoft/deberta-v3-large
```