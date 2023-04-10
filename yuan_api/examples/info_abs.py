import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
set_yuan_account("", "")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是["base_10B","translate","dialog"]之一，"base_10B"是基础模型，"translate"是翻译模型，"dialog"是对话模型
yuan = Yuan(engine="base_10B",
            input_prefix="原文：",
            input_suffix="",
            output_prefix="地点：",
            output_suffix="",
            append_output_prefix_to_query=True,
            max_tokens=128,
            temperature=1,
            topP=0.8,
            topK=1,
            frequencyPenalty=1.0,
            responsePenalty=1.0,
            noRepeatNgramSize=0)

# 3. add examples if in need.
yuan.add_example(Example(inp="明天早晨八点，我从南京北京。",
                        out="南京、北京。"))

print("====信息抽取====")

while(1):
    print("输入Q退出")
    prompt = input("原文：")
    if prompt.lower() == "q":
        break

    response = yuan.submit_API(prompt=prompt, trun="。")
    print(response)
