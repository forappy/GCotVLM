
import requests,json
import pandas as pd
import asyncio
import aiohttp
import json
from tqdm import tqdm
from datetime import datetime
from asyncio import Semaphore
import re
def parser_gpt_response(response_list, **kwargs):
    """
    解析response_list，assistant_dataframe[query, response]
    """
    user_list = []
    assistant_list = []
    for response in response_list:
        user = response[0][-1]
        user_list.append(user)
        try:
            # print(response)
            assistant = response[1]['data']['choices'][0]['content']
            # print(assistant)
            assistant_list.append(assistant)
        except Exception as e:
            print(e)
            assistant_list.append("<|wrong data|>")
    assistant_dataframe = pd.DataFrame()
    assistant_dataframe['query'] = user_list
    assistant_dataframe['response'] = assistant_list
    return assistant_list[0]
class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.current = 0
        self.last_check = datetime.now()
    async def wait(self):
        while True:
            now = datetime.now()
            passed_seconds = (now - self.last_check).seconds
            if passed_seconds > 1:
                self.last_check = now
                self.current = 0
            if self.current < self.rate:
                self.current += 1
                return
            await asyncio.sleep(1)
def make_chat_request_entry(messages, message_type):
    """
    messages: ['',''] .messages[0] = system/user prompt,  message[1] = user/assistant prompt
    示例：
    单轮 system prompt
    # messages = [
    #    {"role": "system", "content": "你是理想同学"}
    #    {"role": "user", "content": "你是谁"},
    # ]
    单轮/多轮 user/assistant prompt
    # messages = [
    #    {"role": "user", "content": "周杰伦的代表作是什么"},
    #    {"role": "assistant", "content": "周杰伦是华语乐坛的重要人物，他的音乐作品多种多样，风格独特，深受广大听众的喜爱。以下是他的一些代表作：\n\n1.《青花瓷》：这首歌是周杰伦的代表作之一，歌曲以中国传统文化为背景，歌词优美，旋律悠扬。\n\n2.《简单爱》：这首歌是周杰伦早期的作品，歌词直接表达了对爱情的向往和追求，深受年轻人的喜爱。\n\n3.《不能说的秘密》：这首歌是周杰伦自编自导的电影《不能说的秘密》的主题曲，歌曲旋律优美，歌词深情。\n\n4.《七里香》：这首歌是周杰伦的经典作品之一，歌曲旋律悠扬，歌词描绘了一段深情的爱情故事。\n\n5.《稻香》：这首歌歌词深入人心，歌曲旋律优美，是周杰伦的代表作之一。\n\n6.《双截棍》：这首歌是周杰伦的代表作之一，歌曲以中国传统武术为主题，展现了周杰伦的音乐创新和实验精神。\n\n7.《夜曲》：这首歌是周杰伦的经典作品之一，歌曲旋律优美，歌词深情，展现了周杰伦的音乐才华。\n\n以上只是周杰伦的部分代表作，他的音乐作品还有很多，每一首都有其独特的魅力。"},
    #    {"role": "user", "content": "他哪一年出道的"}
    # ]
    """
    if message_type == "prompt":
        # 单轮 system prompt
        data_entry = {
            "messages": [{"role": ["system", "user"][i % 2], "content": messages[i]} for i in range(len(messages))],
        }
    else:
        # 单轮/多轮 user/assistant prompt
        data_entry = {
            "messages": [{"role": ["user", "assistant"][i % 2], "content": messages[i]} for i in range(len(messages))],
        }
    data_entry["temperature"] = 0
    return data_entry
async def request_chat_async(url, rate_limiter, semaphore, session, messages, message_type, max_retries=10):
    """
    Async version of the request_chat function
    message_type 表示单轮/多轮/prompt请求
    """
    headers = {'Content-Type': 'application/json'}
    data_entry = make_chat_request_entry(messages, message_type)
    retries = 0
    while retries < max_retries:
        await rate_limiter.wait()  # 控制请求的发出速率
        async with semaphore:  # 限制同时处理的请求数量
            try:
                async with session.post(url, json=data_entry, headers=headers) as response:
                    return await response.json()
            except Exception as e:
                print(f'chatgpt api exception: {e}')
                retries += 1
                await asyncio.sleep(1)
    print('Maximum retry attempts reached, returning error')
    return {"error": "Maximum retry attempts reached, returning error"}
async def process_prompts_chunk_async(url, rate_limiter, semaphore, session, prompts, message_type, max_retries=10):
    """
    Async version of the process_prompts_chunk function
    """
    response = await request_chat_async(url, rate_limiter, semaphore, session, prompts, message_type, max_retries)
    return [prompts, response]
async def gen_assistant_async(url, df, message_type, max_retries=10, qps=2, max_concurrent=20,
                              output_assistant_path="/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/from_gpt.json"):
    """
    qps 最大为2，建议设置小于2
    max_concurrent 为并发数限制，文档没有要求，但是RateLimiter在异步时有时无法控制好qps，因此加此限制，具体数值可根据自身需要调整
    """
    prompts_ls = df
    rate_limiter = RateLimiter(qps)
    semaphore = Semaphore(max_concurrent)  # 限制最大并发数为 max_concurrent，暂时无限制，可以根据自身需求调整大小
    async with aiohttp.ClientSession() as session:
        tasks = [process_prompts_chunk_async(url, rate_limiter, semaphore, session, prompts, message_type, max_retries)
                 for prompts in prompts_ls]
        responses = []
        with open(output_assistant_path, 'a', encoding='utf-8') as f:
            for prompt, future in tqdm(zip(prompts_ls, asyncio.as_completed(tasks)), total=len(tasks)):
                response = await future
                # f.write(json.dumps(response, ensure_ascii=False) + "\n")
                # f.flush()  # ensure the data is written to disk after each iteration
                responses.append(response)
    return responses
def claude_deal_single(url,prompt):
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "messages": [
            {"role": "system", "content": "你是一个中文口语化助手，把以下内容改写成口语化。"},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
            # 检查响应的状态码
        if response.status_code == 200:
            try:
                # 打印响应的内容
                response_json = response.json()  # 假设服务器返回的是JSON格式的内容
                response_result = response_json['data']['choices'][0]['content']
                # print(response_json)
            except json.JSONDecodeError:
                print("响应不是JSON格式：")
                print(response.text)
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)  # 打印详细的错误信息
    except requests.RequestException as e:
        print(f"请求过程中出现异常：{e}")
    return response_result






# # 从GPT回答中提取所需部分
# gpt_responses = []
# entries_with_responses = []
# for i, entry in enumerate(data):
#     for conversation in entry["conversations"]:
#         if conversation["from"] == "gpt":
#             gpt_responses.append(conversation["value"])
#             entries_with_responses.append()
#     if i > 0:
#         break
# data_list = df["text"].values.tolist()
# Please rate the objects you think are important, from 0 to 1. 
# List the noun phrases that are truly directly helpful to the answer
# Combine <CONCLUSION> and select objects that are closely related to the inferred answer.
#  I am providing you with a text containing <CAPTION> and <REASONING> tags.
# prompt = '''You are an expert at analyzing text and identifying key elements that contribute to a deeper understanding of the content.  
#             Your task is to extract noun phrases that represent objects or items which are crucial for understanding the given context and contribute to forming a meaningful conclusion.
#             Remember: 1. Refer to <CONCLUSION> and select objects that are closely related to the inferred answer; 2. When processing the text in the <REASONING> tag, only list the objects that should exist in the image.
#             3. only process <CAPTION> and <REASONING>; 4. If you find more than 4 noun phrases in a tag, sort them by importance and list the top 4 most important ones. 
#             Here are a few examples:
#             Example 1:
#             Input:
#                 <CAPTION>The image shows the Earth's orbit around the Sun, highlighting key dates: the Equinoxes on March 21 and September 22, and the Solstices on June 21 and December 22. The Earth is represented at four positions around its orbit: two equinoxes and two solstices.</CAPTION>
#                 <REASONING>1. Identify the dates of the Solstices from the image: June 21 and December 22.
#                 2. Calculate the interval between June 21 and December 22.
#                 3. Note that from June 21 to December 21 is six months:   - June 21 to July 21: 1 month   - July 21 to August 21: 1 month   - August 21 to September 21: 1 month   - September 21 to October 21: 1 month  - October 21 to November 21: 1 month   - November 21 to December 21: 1 month
#                 Adding these, we have 6 months.\n</REASONING>
#             Output:
#                 <CAPTION>:['the Solstices', 'June 21', 'December 22']
#                 <REASONING>:['June 21', 'December 22']

#             Example 2:
#             Input:
#                 <CAPTION>: The image is a map of the United States, with all states colored light green except for one state in a darker green located in the central part of the country. This darker green state is Illinois. </CAPTION>
#                 <REASONING> The highlighted state on the map, shown in darker green, corresponds to Illinois based on its geographic location in the central U.S. The question asks for the capital of Illinois with multiple-choice options provided. Among the options listed, Springfield is known to be the capital of Illinois. Therefore, the correct answer is Springfield, which matches option D. </REASONING>
#             Output:
#                 <CAPTION>:['United States', 'one state in a darker green']
#                 <REASONING>:['The highlighted state']
            

#             Now, analyze the following text and provide the important and helpful noun phrases in the specified format:
#             text:{}\n
# '''
# result = []

# for text in gpt_responses:
#     prompt_input = prompt.format(text)
#     # loop调用
#     loop = asyncio.get_event_loop()
#     prompt_df = [[prompt_input]]
#     message_type = ""
#     output_assistant_path = '/mnt/pfs-mc0p4k/nlu/team/yuhaofu/data/LLaVA-CoT-100k/from_gpt.json'
#     response_ls = loop.run_until_complete(
#             gen_assistant_async(url, prompt_df, message_type, max_retries=1, qps=2, max_concurrent=20,
#                                 output_assistant_path=output_assistant_path))
#     # response_ls = parser_gpt_response(response_ls).values.tolist()
#     response_ls = parser_gpt_response(response_ls)
#     print(response_ls)
#     result.append(response_ls)


# print(result)
prompt = '''You are an expert at analyzing text and identifying key elements that contribute to a deeper understanding of the content.  
            Your task is to extract noun phrases that represent objects or items which are crucial for understanding the given context and contribute to forming a meaningful conclusion.
            Remember: 1. Refer to <CONCLUSION> and select objects that are closely related to the inferred answer; 2. When processing the text in the <REASONING> tag, only list the objects that should exist in the image.
            3. only process <CAPTION> and <REASONING>; 4. If you find more than 4 noun phrases in a tag, sort them by importance and list the top 4 most important ones. 
            Here are a few examples:
            Example 1:
            Input:
                <CAPTION>The image shows the Earth's orbit around the Sun, highlighting key dates: the Equinoxes on March 21 and September 22, and the Solstices on June 21 and December 22. The Earth is represented at four positions around its orbit: two equinoxes and two solstices.</CAPTION>
                <REASONING>1. Identify the dates of the Solstices from the image: June 21 and December 22.
                2. Calculate the interval between June 21 and December 22.
                3. Note that from June 21 to December 21 is six months:   - June 21 to July 21: 1 month   - July 21 to August 21: 1 month   - August 21 to September 21: 1 month   - September 21 to October 21: 1 month  - October 21 to November 21: 1 month   - November 21 to December 21: 1 month
                Adding these, we have 6 months.\n</REASONING>
            Output:
                <CAPTION>:['the Solstices', 'June 21', 'December 22']
                <REASONING>:['June 21', 'December 22']

            Example 2:
            Input:
                <CAPTION>: The image is a map of the United States, with all states colored light green except for one state in a darker green located in the central part of the country. This darker green state is Illinois. </CAPTION>
                <REASONING> The highlighted state on the map, shown in darker green, corresponds to Illinois based on its geographic location in the central U.S. The question asks for the capital of Illinois with multiple-choice options provided. Among the options listed, Springfield is known to be the capital of Illinois. Therefore, the correct answer is Springfield, which matches option D. </REASONING>
            Output:
                <CAPTION>:['United States', 'one state in a darker green']
                <REASONING>:['The highlighted state']
            

            Now, analyze the following text and provide the important and helpful noun phrases in the specified format:
            text:{}\n
'''
result = []
url = "https://base-model-eval.fc.chj.cloud/gpt4_o"
start_index = 0
end_index = 2

file_path = '/mnt/pfs-mc0p4k/nlu/team/yuhaofu/model/GCotVLM/GCot_json/exceptions.json'
with open(file_path, 'r') as f:
    data = json.load(f)
async def process_conversation(url, text):
    prompt_input = prompt.format(text)
    prompt_df = [[prompt_input]]
    message_type = ""
    # 异步调用接口
    response_ls = await gen_assistant_async(url, prompt_df, message_type, max_retries=1, qps=2, max_concurrent=20)
    # 解析返回结果
    parsed_response = parser_gpt_response(response_ls)
    # print(parsed_response)
 # 尝试处理JSON格式
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        json_match = re.search(json_pattern, parsed_response, re.DOTALL)
    except:
        print(parsed_response)
    if json_match:
        try:
            json_content = json.loads(json_match.group(1))
            # 提取JSON中的内容并重新格式化
            # print(json_content)
            caption = json_content.get("<CAPTION>", "")
            reasoning = json_content.get("<REASONING>", "")
            return f"<CAPTION>:{caption}</CAPTION>\n<REASONING>:{reasoning}</REASONING>"
        except:
            pass
    
    # 检查是否包含冒号分隔的键值对
    caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', parsed_response, re.DOTALL)
    reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', parsed_response, re.DOTALL)

    # caption_content = caption_match.group(1) if caption_match else ""
    # reasoning_content = reasoning_match.group(1) if reasoning_match else ""
    if caption_match and reasoning_match:
        caption_content = caption_match.group(1)
        reasoning_content = reasoning_match.group(1)
    else:
        return parsed_response
    parsed_response = f"<CAPTION>:{caption_content}</CAPTION>\n<REASONING>:{reasoning_content}</REASONING>"
    return parsed_response


async def main(data):
    results = []
    exceptions = []
    save_interval = 1000
    output_path = '/mnt/pfs-mc0p4k/nlu/team/yuhaofu/model/GCotVLM/GCot_json/except_gpt_result.json'
    exceptions_path = '/mnt/pfs-mc0p4k/nlu/team/yuhaofu/model/GCotVLM/GCot_json/test.json'

    # 尝试读取已经处理过的样本
    processed_ids = set()
    try:
        with open(output_path, 'r') as infile:
            existing_results = json.load(infile)
            processed_ids = {entry["id"] for entry in existing_results}
            results.extend(existing_results)
    except (FileNotFoundError, json.JSONDecodeError):
        processed_ids = set()

    # 处理数据
    for entry in data:
        if entry["id"] in processed_ids:
            continue  # 跳过已处理过的样本

        entry_result = {
            "id": entry["id"],
            "image": entry["image"],
            "conversations": []
        }

        try:
            convo_results = []
            for conversation in entry["conversations"]:
                if conversation["from"] == "gpt":
                    processed_result = await process_conversation(url, conversation)
                    convo_results.append({"rec": processed_result})

            entry_result['conversations'] = convo_results
            results.append(entry_result)

        except Exception as e:
            # 记录异常样本
            exceptions.append({
                "id": entry["id"],
                "error": str(e)
            })
            print(f"Error processing entry {entry['id']}: {e}")

        # 定期保存结果到JSON文件
        if len(results) % save_interval == 0:
            with open(output_path, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            print(f"Saved progress; processed up to entry id {entry['id']}.")

    # 保存最终结果到JSON文件
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    # 保存异常样本到单独的JSON文件
    with open(exceptions_path, 'w') as errfile:
        json.dump(exceptions, errfile, indent=4)

    print(f"Processing complete with {len(exceptions)} exceptions.")
loop = asyncio.get_event_loop()
loop.run_until_complete(main(data))