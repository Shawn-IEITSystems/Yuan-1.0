import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
yuan = Yuan(input_prefix="以",
            input_suffix="为题作一首诗：",
            output_prefix="答：",
            output_suffix="”",
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="清风",
                        out="春风用意匀颜色，销得携觞与赋诗。秾丽最宜新著雨，娇饶全在欲开时。"))
yuan.add_example(Example(inp="",out="我知道，不需要，是免费开放。"))
# yuan.add_example(Example(inp="你还可以到它边上的观复博物馆看看，我觉得那里很不错。",out="观复博物馆我知道，是马未都先生创办的新中国第一家私立博物馆。"))

print("====作诗机器人====")

while(1):
    print("输入Q退出")
    prompt = input("以何为题作诗：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="”")
    print(response+"”")
# # 4. get response
# prompt = "故宫的珍宝馆里有什么好玩的？"
# response = yuan.submit_API(prompt=prompt,trun="”")

# print(prompt)
# print(response+"”")