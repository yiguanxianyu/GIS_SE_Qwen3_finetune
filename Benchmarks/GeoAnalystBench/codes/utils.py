import asyncio
import csv
import re

import pandas as pd
from openai import AsyncOpenAI


def extract_task_list(long_string):
    """Extract the task list from a long string."""
    lines = long_string.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not re.match(r"^\d+\.", line)]
    return lines


def create_client(api_key=None, base_url=None):
    """
    创建一个OpenAI兼容的异步客户端

    Args:
        api_key: API密钥
        base_url: API基础URL (例如: "https://dashscope.aliyuncs.com/compatible-mode/v1")

    Returns:
        AsyncOpenAI客户端实例
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client


async def call_llm(
    prompt, model, temperature=0.7, max_tokens=None, timeout=None, api_key=None, base_url=None, remove_thinking=True
):
    """
    异步通用的LLM调用函数

    Args:
        prompt: 提示词
        model: 模型名称 (例如: "gpt-4o-mini", "qwen-plus", "deepseek-chat")
        temperature: 温度参数
        max_tokens: 最大token数
        timeout: 超时时间
        api_key: API密钥
        base_url: API基础URL
        remove_thinking: 是否移除思维链标签 (针对 deepseek-r1 等模型)

    Returns:
        模型响应内容
    """
    client = create_client(api_key=api_key, base_url=base_url)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        result = response.choices[0].message.content

        # 处理思维链模型的输出
        if remove_thinking and "</think>" in result:
            return result.split("</think>")[1].strip()

        return result
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise


def calculate_workflow_length_loss(annotations, responses):
    loss = 0
    for i in range(len(annotations)):
        filtered_responses = responses[responses["task_id"] == annotations.iloc[i]["id"]]
        for j in range(len(filtered_responses)):
            loss += abs(filtered_responses.iloc[j]["task_length"] - annotations.iloc[i]["task_length"])
    loss = loss / len(annotations)
    return loss


def find_task_length(outline):
    number_list = []
    lines = outline.split("\n")
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char.isdigit():
                # Check if next char forms a 2-digit number
                if j + 1 < len(line) and line[j + 1].isdigit():
                    if j + 2 < len(line) and line[j + 2] == ".":
                        num = int(line[j : j + 2])
                        number_list.append(num)
                else:
                    if j + 1 < len(line) and line[j + 1] == ".":
                        num = int(char)
                        if num <= 10:
                            number_list.append(num)
    if not number_list:
        return 0
    return max(number_list)


async def process_single_prompt(
    prompt_data, api_type, model, temperature, api_key, base_url, remove_thinking, num_responses=3
):
    """
    处理单个提示词,调用API多次并返回结果

    Args:
        prompt_data: 包含提示词和元数据的数据行
        api_type: 'workflow' or 'code'
        model: 模型名称
        temperature: 温度参数
        api_key: API密钥
        base_url: API基础URL
        remove_thinking: 是否移除思维链标签
        num_responses: 每个提示词调用API的次数

    Returns:
        包含所有响应的列表
    """
    prompt = prompt_data["prompt_content"]

    # 创建多个异步任务,并发调用API
    tasks = [
        call_llm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            remove_thinking=remove_thinking,
        )
        for _ in range(num_responses)
    ]

    # 等待所有任务完成
    responses = await asyncio.gather(*tasks)

    # 准备写入的数据
    if prompt_data["domain_knowledge"] and prompt_data["dataset"]:
        prompt_type = "domain_and_dataset"
    elif prompt_data["domain_knowledge"]:
        prompt_type = "domain"
    elif prompt_data["dataset"]:
        prompt_type = "dataset"
    else:
        prompt_type = "original"

    rows = []
    for k, response in enumerate(responses):
        if api_type == "workflow":
            task_length = len(extract_task_list(response))
        else:
            task_length = "none"

        rows.append(
            [
                prompt_data["task_id"],
                str(prompt_data["task_id"]) + api_type + str(k),
                prompt_type,
                api_type,
                prompt_data["Arcpy"],
                model,
                response,
                task_length,
            ]
        )

    return rows


async def call_api_async(
    api_type,
    prompt_file,
    output_file,
    model,
    temperature=0.7,
    api_key=None,
    base_url=None,
    remove_thinking=True,
    max_concurrent=5,
):
    """
    异步调用API进行批量推理

    Args:
        api_type: 'workflow' or 'code'
        prompt_file: CSV文件路径,包含所有提示词
        output_file: CSV文件路径,用于写入响应结果
        model: 模型名称 (例如: "gpt-4o-mini", "qwen-plus", "deepseek-chat")
        temperature: 温度参数
        api_key: API密钥
        base_url: API基础URL
        remove_thinking: 是否移除思维链标签
        max_concurrent: 最大并发数

    该函数会对prompt_file中的每个提示词调用API,并将响应写入output_file
    """
    api_key = "sk-cnktibhdyfwbasbpbkaloheuhccqftlcrqrfcmpniupwbwuv"
    base_url = "https://api.siliconflow.cn/v1"
    prompts = pd.read_csv(prompt_file)

    # 写入CSV头
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "response_id",
                "prompt_type",
                "response_type",
                "Arcpy",
                "llm_model",
                "response_content",
                "task_length",
            ]
        )

    # 使用信号量控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(idx, row):
        async with semaphore:
            print(f"\r处理中: {idx + 1}/{len(prompts)}", end="", flush=True)
            return await process_single_prompt(row, api_type, model, temperature, api_key, base_url, remove_thinking)

    # 创建所有任务
    tasks = [process_with_semaphore(i, row) for i, (_, row) in enumerate(prompts.iterrows())]

    # 并发执行所有任务
    all_results = await asyncio.gather(*tasks)

    # 写入所有结果
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        for results in all_results:
            for row in results:
                writer.writerow(row)

    print("\n完成!")


def call_api(
    api_type,
    prompt_file,
    output_file,
    model,
    temperature=0.7,
    api_key=None,
    base_url=None,
    remove_thinking=True,
    max_concurrent=5,
):
    """
    调用API进行批量推理 (同步包装器)

    Args:
        api_type: 'workflow' or 'code'
        prompt_file: CSV文件路径,包含所有提示词
        output_file: CSV文件路径,用于写入响应结果
        model: 模型名称 (例如: "gpt-4o-mini", "qwen-plus", "deepseek-chat")
        temperature: 温度参数
        api_key: API密钥
        base_url: API基础URL
        remove_thinking: 是否移除思维链标签
        max_concurrent: 最大并发数

    该函数会对prompt_file中的每个提示词调用API,并将响应写入output_file
    """
    asyncio.run(
        call_api_async(
            api_type, prompt_file, output_file, model, temperature, api_key, base_url, remove_thinking, max_concurrent
        )
    )


if __name__ == "__main__":
    model = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # call_api(
    #     "code",
    #     "/Users/gx/Documents/Source/GIS_SE_Qwen3_finetune/GeoAnalystBench/codes/code_prompts.csv",
    #     f"GeoAnalystBench/response/code_responses_{model.split('/')[-1]}.csv",
    #     model,
    # )
    call_api(
        "workflow",
        "/Users/gx/Documents/Source/GIS_SE_Qwen3_finetune/GeoAnalystBench/codes/workflow_prompts.csv",
        f"GeoAnalystBench/response/workflow_responses_{model.split('/')[-1]}.csv",
        model,
    )
