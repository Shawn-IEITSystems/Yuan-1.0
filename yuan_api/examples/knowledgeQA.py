import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='dialog',
            input_prefix="问：",
            input_suffix="",
            output_prefix="答：",
            output_suffix="。",
            temperature=0.1,
            topP=1,
            topK=5,
            max_tokens=128,
            append_output_prefix_to_query=True)

# 3. add examples if in need.

print("====知识问答====")

while(1):
    print("输入Q退出")
    prompt = input("问：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="。")
    print("答："+response+"。")
