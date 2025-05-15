# Text2SQL 自然语言转SQL查询系统

SCNU 2025Spring NLP课程比赛项目

## 功能特点

- 支持自然语言到SQL的转换
- 模型：Google Gemini API
- 批量处理能力
- 异步处理支持
- 完善的错误处理和重试机制
- 结果缓存功能

## 系统要求

- Python 3.8+
- CUDA支持（推荐，用于GPU加速）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建并激活虚拟环境：
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
.
├── model.py          # 模型定义和实现
├── train.py          # 训练脚本
├── requirements.txt  # 项目依赖
├── text2sql/         # 数据目录
│   ├── train.json    # 训练数据
│   └── test.json     # 测试数据
└── result/           # 结果输出目录
```

## 使用方法

1. 准备数据：
   - 将训练数据放在 `text2sql/train.json`
   - 将测试数据放在 `text2sql/test.json`

2. 运行训练：
```bash
python train.py
```

3. 查看结果：
   - 预测结果将保存在 `result/submission.csv` 文件中

## 配置说明

在 `model.py` 中可以配置以下参数：
- API密钥
- 批处理大小
- 重试策略
- 超时设置

## 注意事项

1. 确保有足够的API调用额度
2. 建议使用GPU进行训练
3. 注意数据格式要求：
   - 训练数据需要包含 'NL'（自然语言）和 'SQL'（SQL查询）字段
   - 测试数据需要包含 'NL' 和 'id' 字段

## 依赖版本

- torch==1.13.1
- transformers==4.26.1
- pandas==1.5.3
- numpy==1.24.3
- tqdm==4.65.0
- huggingface-hub==0.12.1
- accelerate==0.16.0
- einops==0.6.1
- protobuf==3.20.0
