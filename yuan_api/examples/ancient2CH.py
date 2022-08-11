import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='rhythm_poems',
            input_prefix="将下列古文翻译成现代文。古文：",
            input_suffix="",
            output_prefix="现代文：",
            output_suffix="",
            topK=5,
            temperature=1,
            max_tokens=50,
            topP=0.9,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="臣闻求木之长者，必固其根本；欲流之远者，必浚其泉源；思国之安者，必积其德义。",
                        out="臣听说要求树木长得高大，一定要稳固它的根底；想要河水流得远长，一定要疏通它的源泉；要使国家安定，一定要积聚它的德义。"))

print("====古文翻译====")

while(1):
    print("输入Q退出")
    prompt = input("古文：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="")
    print(response+"")
