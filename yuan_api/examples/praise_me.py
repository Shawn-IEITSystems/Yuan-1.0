import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='dialog',
            input_prefix="问：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            append_output_prefix_to_query=True,
            topK=5,
            temperature=1,
            topP=0.8,
            frequencyPenalty=1.2)

# 3. add examples if in need.
yuan.add_example(Example(inp="夸我",
                        out="从您的言谈中可以看出，我今天遇到的是很有修养的人。"))
yuan.add_example(Example(inp="已经上年纪了，忧桑。",out="别开玩笑了，看您的容貌，肯定不到二十岁。"))
yuan.add_example(Example(inp="被老板怼了，求夸。",out="您一看就是大富大贵的人，在同龄人中，您的能力真是出类拔萃！"))

print("====夸夸机器人====")

while(1):
    print("输入Q退出")
    prompt = input("我：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="”")
    print('yuan：“' + response+"”")
