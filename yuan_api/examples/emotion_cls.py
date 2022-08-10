import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            input_prefix="客户评价",
            input_suffix="",
            output_prefix="结论：",
            output_suffix="。",
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="请对客户评价做出情感判别，给出好评还是差评的结论。",
                        out=""))
yuan.add_example(Example(inp="第一次在网上买东西，结果就买到与样品不符合的衣服，纠结死我了。短裤变长裤，男裤变女裤。这叫我情何以堪。",
                        out="差评。"))

print("====情感分类====")

while(1):
    print("输入Q退出")
    prompt = input("客户评价：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="。")
    print(response+"。")
