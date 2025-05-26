import json
import csv
import zipfile
import asyncio
from vllm import AsyncLLM, SamplingParams

# 你的模型路径或huggingface模型名，改成实际路径
MODEL_NAME = "adapter_model.safetensors"

# 测试数据文件名（json）
TEST_FILE = "test_with_input.json"

# 输出csv和压缩包文件名
CSV_FILE = "submission.csv"
ZIP_FILE = "submission.zip"

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 根据你的测试数据结构定制 prompt，这里假设你想用instruction和input字段
def build_prompt(entry):
    # 这里你可以根据微调的prompt格式调整
    return (
        f"Instruction: {entry['instruction']}\n"
        f"Input: {entry['input']}\n"
        "请生成对应的SQL查询语句：\nSQL:"
    )

async def main():
    data = load_data(TEST_FILE)

    llm = AsyncLLM(MODEL_NAME)
    sampling_params = SamplingParams(temperature=0, max_tokens=200)

    results = []

    tasks = []
    for entry in data:
        prompt = build_prompt(entry)
        task = llm.generate(prompt, sampling_params)
        tasks.append((entry["id"], task))

    for id_, task in tasks:
        result = await task
        pred_sql = result.generations[0].text.strip()
        results.append((id_, pred_sql))

    # 写入CSV，UTF-8编码
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "pred_sql"])
        writer.writerows(results)

    # 压缩CSV文件
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(CSV_FILE)

    print(f"完成！生成的文件：{CSV_FILE} 和压缩包 {ZIP_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
