import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='rhythm_poems',
            input_prefix="以",
            input_suffix="为题作一首诗：",
            output_prefix="答：",
            output_suffix="”",
            max_tokens=40,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="清风",
                        out="春风用意匀颜色，销得携觞与赋诗。秾丽最宜新著雨，娇饶全在欲开时。"))

print("====作诗机器人====")

while(1):
    print("输入Q退出")
    prompt = input("以何为题作诗：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="”")
    print(response+"”")

# # 4. get response
# prompt = "清风"
# response = yuan.submit_API(prompt=prompt,trun="”")
# print(response+"”")
