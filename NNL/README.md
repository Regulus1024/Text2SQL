# Text2SQL 自然语言转SQL查询系统

这是一个基于深度学习的自然语言转SQL查询（Text2SQL）系统，可以将自然语言问题转换为对应的SQL查询语句。

## 项目结构

```
NNL/
├── data/               # 数据目录
├── src/               # 源代码目录
│   ├── config.py      # 配置文件
│   ├── data/          # 数据处理相关代码
│   └── model/         # 模型相关代码
├── train.py           # 训练脚本
├── predict.py         # 预测脚本
├── check_environment.py # 环境检查脚本
└── requirements.txt   # 项目依赖
```

## 环境要求

- Python 3.7+
- CUDA 11.0+ (用于GPU加速)
- 8GB+ RAM
- 10GB+ 磁盘空间

## 安装步骤

1. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 检查环境配置：
```bash
python check_environment.py
```

## 使用方法

### 训练模型

1. 准备训练数据：
   - 将训练数据放在 `data/train/` 目录下
   - 数据格式应为JSON文件，包含自然语言问题和对应的SQL查询

2. 开始训练：
```bash
python train.py
```

训练过程中会：
- 自动创建日志文件 `training.log`
- 保存最佳模型到 `checkpoints/` 目录
- 使用wandb进行训练监控（可选）

### 生成预测

1. 准备测试数据：
   - 将测试数据放在 `data/test/` 目录下
   - 数据格式应为JSON文件，包含自然语言问题

2. 生成预测：
```bash
python predict.py
```

预测结果将保存在：
- `output/submission/submission.csv`
- `output/submission/submission.zip`

## 主要特性

- 基于Transformer架构的深度学习模型
- 支持批量处理和预测
- 自动环境检查和配置
- 完整的日志记录
- 模型检查点保存
- 支持GPU加速

## 依赖包版本

- torch==1.13.1
- transformers==4.26.1
- pandas==1.3.0
- numpy==1.19.5
- tqdm==4.62.0
- scikit-learn==0.24.2
- sqlparse==0.4.2
- python-dotenv==0.19.0
- accelerate==0.12.0
- psutil==5.8.0
- wandb==0.12.0

## 注意事项

1. 首次运行前请确保已正确配置环境变量
2. 训练过程可能需要较长时间，请确保有足够的计算资源
3. 建议使用GPU进行训练
4. 定期备份模型检查点

## 常见问题

1. 如果遇到CUDA相关错误，请检查GPU驱动和CUDA版本
2. 如果内存不足，可以调整batch_size参数
3. 如果训练速度较慢，可以尝试使用更小的模型或减少训练数据量

## 许可证

MIT License