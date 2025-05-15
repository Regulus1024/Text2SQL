import os
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from src.config import Config
from src.data.processor import Text2SQLDataProcessor
from src.model.text2sql_model import Text2SQLModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def main():
    # 初始化配置
    config = Config()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建必要的目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 初始化数据处理器
    data_processor = Text2SQLDataProcessor(config)
    
    # 加载数据
    train_data = data_processor.load_data(config.train_data_path)
    train_data, val_data = data_processor.split_data(train_data)
    
    # 创建数据加载器
    train_loader = data_processor.create_dataloader(train_data, is_training=True)
    val_loader = data_processor.create_dataloader(val_data, is_training=False)
    
    # 初始化模型
    device = torch.device(config.device)
    model = Text2SQLModel(config).to(device)
    
    # 初始化优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练模型
    model.train_model(train_loader, val_loader, optimizer, scheduler)

if __name__ == "__main__":
    main() 