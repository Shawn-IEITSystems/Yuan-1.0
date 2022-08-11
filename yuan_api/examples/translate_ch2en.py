import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
# 本示例构建中翻英，用户可对照修改自行构建英翻中。
yuan = Yuan(engine='translate',
            input_prefix="将下列中文翻译成英文。中文：",
            input_suffix="",
            output_prefix="英文：",
            output_suffix="",
            append_output_prefix_to_query=True)

# 3. add examples if in need.
# 翻译机器人zeroshot效果最好
# yuan.add_example(Example(inp="登陆月球开创了太空探索的新纪元。",
#                         out="The moon landing inaugurated a new era in space exploration."))

print("====中翻英机器人====")

while(1):
    print("输入Q退出")
    prompt = input("中文：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="")
    print(response)
