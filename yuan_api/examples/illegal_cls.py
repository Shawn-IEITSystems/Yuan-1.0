import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            input_prefix="内容",
            input_suffix="",
            output_prefix="答案是：",
            output_suffix="。",
            temperature=0.1,
            topK=5,
            max_tokens=10,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="我想和你滚床单。这句话包含黄色信息吗？",
                        out="包含。"))
yuan.add_example(Example(inp="西湖美景盖世无双，我想在这里多待几天。这句话包含黄色信息吗？",
                        out="不包含。"))

print("====非法鉴别====")

while(1):
    print("输入Q退出")
    prompt = input("内容：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="。")
    print(response+"。")
