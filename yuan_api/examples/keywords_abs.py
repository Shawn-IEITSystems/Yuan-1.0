import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            input_prefix="从下面的段落中抽取关键词：",
            input_suffix="",
            output_prefix="关键词：",
            output_suffix="。",
            topK=5,
            temperature=0.1,
            max_tokens=10,
            topP=0.9,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="关键词检测是语音识别领域的一个子领域，其目的是在语音信号中检测指定词语的所有出现位置。",
                        out="关键词，语音识别，位置。"))


print("====关键词生成====")

while(1):
    print("输入Q退出")
    prompt = input("从下面的段落中抽取关键词")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="。")
    print(response)
