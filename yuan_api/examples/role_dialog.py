import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='dialog',
            input_prefix="人：“",
            input_suffix="",
            output_prefix="AI：“",
            output_suffix="”",
            append_output_prefix_to_query=True,
            topK=5,
            temperature=0.9,
            topP=0.8,
            max_tokens=40,
            frequencyPenalty=1.2)

# 3. add examples if in need.
yuan.add_example(Example(inp="人：该AI机器人的姓名为小源、性别女，年龄18岁，出生地北京，受教育情况高中，身高168cm，喜好唱歌和旅游。",out=""))

print("====人物对话====")

while(1):
    print("输入Q退出")
    prompt = input("人：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="”")
    print(response+"”")
