import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aiohttp
import time
from tqdm import tqdm
import requests

class Text2SQLDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        try:
            question, sql = self.data[idx]
            
            # 构建提示模板
            prompt = f"请将以下自然语言问题转换为SQL查询：\n问题：{question}\nSQL："
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 编码目标
            targets = self.tokenizer(
                sql,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            }
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }

class Text2SQLModel:
    def __init__(self, api_key: str = "AIzaSyCrEx0wJfZGRJotekFXWiu2Zd4pY4XY80M", 
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        print(f"初始化模型，使用API地址: {base_url}")
        self.api_key = api_key
        self.base_url = base_url
        self.cache = {}  # 用于存储API调用结果
        self.last_call_time = 0  # 用于控制API调用频率
        self.min_interval = 1.0  # 增加最小调用间隔到1秒
        self.batch_size = 100  # 每批处理的问题数量
        print("模型初始化完成")
    
    def train(self, train_data: List[Tuple[str, str]], 
              batch_size: int = 2,
              epochs: int = 3,
              learning_rate: float = 1e-5):
        """由于使用API，这里只记录训练数据，不进行实际训练"""
        print(f"记录训练数据，共 {len(train_data)} 条")
        self.train_data = train_data
    
    def _build_prompt(self, questions: List[str]) -> str:
        """构建批量处理的提示词"""
        prompt = """你是SQL高级工程师,能熟练将自然语言转化为SQL处理,请将以下自然语言问题转换为SQL查询。
重要说明：
1. 必须为每个问题生成一个SQL查询
2. 每个SQL查询必须单独一行
3. 严格按照问题顺序输出结果
4. 不要添加任何额外的解释或说明
5. 确保输出行数与问题数量相同
6. 每个SQL查询必须完整，不能截断
7. 如果无法生成SQL，请输出 SELECT NULL;

问题列表：
"""
        for i, question in enumerate(questions, 1):
            prompt += f"{i}. {question}\n"
        
        prompt += f"\n请直接输出{len(questions)}个SQL查询，每行一个，不要添加任何其他内容。"
        return prompt
    
    def _parse_responses(self, response: str, num_questions: int) -> List[str]:
        """解析批量响应"""
        # 按行分割响应，过滤空行
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # 处理每一行，确保是有效的SQL
        predictions = []
        for line in lines:
            # 移除可能的SQL标记
            line = line.replace('```sql', '').replace('```', '').strip()
            
            # 如果行不是以SELECT开头，添加SELECT
            if not line.upper().startswith('SELECT'):
                line = f"SELECT {line}"
            
            # 确保SQL语句完整
            if not line.endswith(';'):
                line += ';'
                
            predictions.append(line)
        
        # 如果预测结果数量不够，用默认值补充
        while len(predictions) < num_questions:
            predictions.append("SELECT NULL;")
            
        # 如果预测结果数量太多，只取需要的数量
        return predictions[:num_questions]
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _call_api(self, prompt: str) -> str:
        """带重试机制的API调用"""
        print(f"开始API调用，提示词长度: {len(prompt)}")
        # 控制调用频率
        current_time = time.time()
        if current_time - self.last_call_time < self.min_interval:
            sleep_time = self.min_interval - (current_time - self.last_call_time)
            print(f"等待 {sleep_time:.2f} 秒后继续...")
            time.sleep(sleep_time)
        
        try:
            print("发送API请求...")
            url = f"{self.base_url}/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,  # 降低温度以获得更稳定的输出
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 8192  # 增加最大输出长度
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"API调用成功，响应长度: {len(response_text)}")
                self.last_call_time = time.time()
                return response_text
            else:
                raise Exception("API响应格式不正确")
                
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            raise  # 让重试装饰器处理异常

    def predict(self, questions: List[str], batch_size: int = None) -> List[str]:
        """批量预测方法"""
        print(f"开始预测，总问题数量: {len(questions)}")
        
        if batch_size is None:
            batch_size = self.batch_size
            
        all_predictions = []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]
            
            print(f"\n处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch_questions)} 个问题")
            
            try:
                # 构建批量提示词
                prompt = self._build_prompt(batch_questions)
                
                # 调用API
                response = self._call_api(prompt)
                
                # 解析响应
                batch_predictions = self._parse_responses(response, len(batch_questions))
                
                # 验证结果数量
                if len(batch_predictions) != len(batch_questions):
                    print(f"警告：预测结果数量({len(batch_predictions)})与问题数量({len(batch_questions)})不匹配")
                    # 确保结果数量正确
                    if len(batch_predictions) < len(batch_questions):
                        batch_predictions.extend(["SELECT NULL;"] * (len(batch_questions) - len(batch_predictions)))
                    else:
                        batch_predictions = batch_predictions[:len(batch_questions)]
                
                all_predictions.extend(batch_predictions)
                print(f"当前批次完成，已生成 {len(all_predictions)}/{len(questions)} 个结果")
                
            except Exception as e:
                print(f"处理批次 {batch_idx + 1} 时出错: {str(e)}")
                # 如果出错，用默认值补充
                all_predictions.extend(["SELECT NULL;"] * len(batch_questions))
            
            # 批次间增加延迟
            if batch_idx < total_batches - 1:
                time.sleep(2)
        
        print(f"\n所有批次处理完成，共生成 {len(all_predictions)} 个结果")
        return all_predictions

    def predict_fast(self, questions: List[str], batch_size: int = None) -> List[str]:
        """为了保持接口兼容性，保留此方法"""
        return self.predict(questions, batch_size)

    # 新增优化版本的预测方法
    async def _call_api_async(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """异步API调用"""
        try:
            print(f"开始异步API调用，提示词长度: {len(prompt)}")
            # 控制调用频率
            current_time = time.time()
            if current_time - self.last_call_time < self.min_interval:
                sleep_time = self.min_interval - (current_time - self.last_call_time)
                print(f"等待 {sleep_time:.2f} 秒后继续...")
                await asyncio.sleep(sleep_time)
            
            print("发送异步API请求...")
            url = f"{self.base_url}/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            async with session.post(
                url,
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=30  # 增加超时时间
            ) as response:
                print(f"收到响应，状态码: {response.status}")
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"触发限流，等待 {retry_after} 秒后重试...")
                    await asyncio.sleep(retry_after)
                    return await self._call_api_async(session, prompt)
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"API调用失败: HTTP {response.status} - {error_text}")
                    raise Exception(f"API调用失败: HTTP {response.status} - {error_text}")
                
                try:
                    result = await response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        print(f"异步API调用成功，响应长度: {len(response_text)}")
                        self.last_call_time = time.time()
                        return response_text
                    else:
                        raise Exception("API响应格式不正确")
                except Exception as e:
                    print(f"解析响应失败: {str(e)}")
                    raise
                    
        except Exception as e:
            print(f"异步API调用出错: {str(e)}")
            raise

    async def predict_async(self, questions: List[str], batch_size: int = 2) -> List[str]:
        """异步批量预测方法"""
        print(f"开始异步预测，问题数量: {len(questions)}")
        predictions = []
        
        # 分批处理问题
        for i in tqdm(range(0, len(questions), batch_size), desc="处理批次"):
            batch_questions = questions[i:i + batch_size]
            print(f"\n处理第 {i//batch_size + 1} 批，包含 {len(batch_questions)} 个问题")
            
            batch_prompts = [
                f"请将以下自然语言问题转换为SQL查询：\n问题：{q}\nSQL："
                for q in batch_questions
            ]
            
            # 检查缓存
            cached_results = [self.cache.get(prompt, None) for prompt in batch_prompts]
            need_api_prompts = [p for p, cached in zip(batch_prompts, cached_results) if cached is None]
            print(f"需要API调用的提示词数量: {len(need_api_prompts)}")
            
            if need_api_prompts:
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for prompt in need_api_prompts:
                        try:
                            task = asyncio.create_task(self._call_api_async(session, prompt))
                            tasks.append(task)
                        except Exception as e:
                            print(f"创建任务失败: {str(e)}")
                            tasks.append(None)
                    
                    new_results = []
                    for task in tasks:
                        if task:
                            try:
                                result = await task
                                new_results.append(result)
                            except Exception as e:
                                print(f"任务执行失败: {str(e)}")
                                new_results.append("")
                        else:
                            new_results.append("")
                    
                    # 更新缓存
                    for prompt, result in zip(need_api_prompts, new_results):
                        if result:  # 只缓存非空结果
                            self.cache[prompt] = result
            
            # 收集所有结果
            batch_predictions = []
            for prompt in batch_prompts:
                if prompt in self.cache:
                    sql = self._process_response(self.cache[prompt])
                else:
                    sql = ""
                batch_predictions.append(sql)
            
            predictions.extend(batch_predictions)
            print(f"当前批次处理完成，已生成 {len(batch_predictions)} 个预测结果")
            
            # 批次间增加延迟
            await asyncio.sleep(1)
        
        print(f"预测完成，共生成 {len(predictions)} 个结果")
        return predictions