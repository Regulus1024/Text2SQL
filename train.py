import os
import json
import pandas as pd
from model import Text2SQLModel
import torch
from tqdm import tqdm

def load_json_file(file_path: str) -> list:
    """加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"加载文件 {file_path} 失败: {str(e)}")

def extract_train_fields(item: dict) -> tuple:
    """提取训练数据字段
    
    Args:
        item: 数据项
        
    Returns:
        (问题, SQL查询)元组
    """
    question = item.get('NL')
    sql = item.get('SQL')
    if not question or not sql:
        raise ValueError(f"训练数据缺少必要字段，数据项: {item}")
    return question, sql

def extract_test_fields(item: dict) -> tuple:
    """提取测试数据字段
    
    Args:
        item: 数据项
        
    Returns:
        (问题, ID)元组
    """
    question = item.get('NL')
    item_id = item.get('id')
    if not question or not item_id:
        raise ValueError(f"测试数据缺少必要字段，数据项: {item}")
    return question, item_id

def prepare_data(train_file: str, test_file: str) -> tuple:
    """准备训练和测试数据
    
    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
        
    Returns:
        (训练数据列表, 测试数据列表)
    """
    # 加载数据
    train_data = load_json_file(train_file)
    test_data = load_json_file(test_file)
    
    # 提取字段
    train_processed = [extract_train_fields(item) for item in train_data]
    test_processed = [extract_test_fields(item) for item in test_data]
    
    return train_processed, test_processed

def save_predictions(predictions: list, test_data: list, output_file: str):
    """保存预测结果
    
    Args:
        predictions: 预测的SQL查询列表
        test_data: 测试数据
        output_file: 输出文件路径
    """
    # 创建结果DataFrame
    results = pd.DataFrame({
        'id': [item['id'] for item in test_data],
        'pred_sql': predictions
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存为CSV
    results.to_csv(output_file, index=False, encoding='utf-8')

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 数据文件路径
    train_file = 'text2sql/train.json'
    test_file = 'text2sql/test.json'
    
    print(f"训练数据文件: {train_file}")
    print(f"测试数据文件: {test_file}")
    
    # 创建输出目录
    os.makedirs('result', exist_ok=True)
    print("创建输出目录: result")
    
    # 准备数据
    print("\n正在加载数据...")
    try:
        train_data, test_data = prepare_data(train_file, test_file)
        print(f"成功加载训练数据: {len(train_data)} 条")
        print(f"成功加载测试数据: {len(test_data)} 条")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return
    
    # 加载原始测试数据（用于保存ID）
    try:
        test_raw = load_json_file(test_file)
        print(f"成功加载原始测试数据: {len(test_raw)} 条")
    except Exception as e:
        print(f"加载原始测试数据失败: {str(e)}")
        return
    
    # 初始化模型
    print("\n正在初始化模型...")
    try:
        model = Text2SQLModel(api_key="AIzaSyCrEx0wJfZGRJotekFXWiu2Zd4pY4XY80M")
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return
    
    # 训练模型
    print("\n开始训练...")
    try:
        model.train(
            train_data=train_data,
            batch_size=2,
            epochs=3,
            learning_rate=1e-5
        )
        print("训练完成")
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        return
    
    # 生成预测
    print("\n正在生成预测...")
    try:
        test_questions = [item[0] for item in test_data]  # 只取问题部分
        print(f"准备预测 {len(test_questions)} 个问题")
        predictions = model.predict_fast(test_questions, batch_size=2)
        print(f"成功生成 {len(predictions)} 个预测结果")
    except Exception as e:
        print(f"生成预测失败: {str(e)}")
        return
    
    # 保存预测结果
    print("\n正在保存预测结果...")
    try:
        save_predictions(predictions, test_raw, 'result/submission.csv')
        print("预测结果已保存到 result/submission.csv")
    except Exception as e:
        print(f"保存预测结果失败: {str(e)}")
        return
    
    print("\n所有任务完成！")

if __name__ == '__main__':
    main() 