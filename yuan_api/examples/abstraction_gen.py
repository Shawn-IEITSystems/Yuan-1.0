import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            input_prefix="内容：",
            input_suffix="",
            output_prefix="文章标题：",
            output_suffix="。",
            topK=5,
            temperature=0.1,
            max_tokens=30,
            topP=0.6,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="截至5月27日，今年西部陆海新通道海铁联运班列开行3173列，累计约15.9万标箱，同比增长33%，完成上半年开行3150列的阶段性任务目标，提前1个月完成“双过半”任务。今年前4月，RCEP成员国经西部陆海新通道发运22111标箱，占通道到发总运量17.6%，外贸到发运量51482标箱，同比增长58.5%。",
                        out="西部陆海新通道海铁联运班列提前完成上半年目标任务。"))

print("====摘要生成====")

while(1):
    print("输入Q退出")
    prompt = input("内容：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="。")
    print(response+"。")
