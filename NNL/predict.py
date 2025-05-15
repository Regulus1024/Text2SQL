import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.config import Config
from src.data.processor import Text2SQLDataProcessor
from src.model.text2sql_model import Text2SQLModel

def main():
    # 初始化配置
    config = Config()
    
    # 设置设备
    device = torch.device(config.device)
    
    # 初始化数据处理器
    data_processor = Text2SQLDataProcessor(config)
    
    # 加载测试数据
    test_data = data_processor.load_data(config.test_data_path)
    
    # 创建测试数据加载器
    test_loader = data_processor.create_dataloader(test_data, is_training=False)
    
    # 初始化模型
    model = Text2SQLModel(config).to(device)
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(f"{config.checkpoint_dir}/best_model.pt"))
    model.eval()
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 存储预测结果
    predictions = []
    
    # 生成预测
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']
            
            # 生成SQL
            generated_ids = model.generate(input_ids, attention_mask)
            
            # 解码生成的SQL
            generated_sqls = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 保存预测结果
            for id_, sql in zip(ids, generated_sqls):
                predictions.append({
                    'id': id_,
                    'pred_sql': sql
                })
    
    # 创建输出目录
    os.makedirs('output/submission', exist_ok=True)
    
    # 将预测结果转换为DataFrame
    df = pd.DataFrame(predictions)
    
    # 保存为CSV文件
    output_path = 'output/submission/submission.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 压缩文件
    import zipfile
    with zipfile.ZipFile('output/submission/submission.zip', 'w') as zipf:
        zipf.write(output_path, arcname='submission.csv')
    
    print(f"预测结果已保存到: {output_path}")
    print(f"压缩文件已保存到: output/submission/submission.zip")

if __name__ == "__main__":
    main() 