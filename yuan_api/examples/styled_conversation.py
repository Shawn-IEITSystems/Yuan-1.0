import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            input_prefix="",
            input_suffix="",
            output_prefix="",
            output_suffix="",
            topK=5,
            temperature=0.9,
            topP=0.9,
            max_tokens=200,
            append_output_prefix_to_query=False,
            frequencyPenalty=1.2)

# 3. add examples if in need.
yuan.add_example(Example(inp="提起她，脑海中不禁浮现两个字：霸气。因此，在背后我常常称呼哪位颇具霸气风范的女孩为“御姐”。她个子不高，性子冷。鼻梁上架着一副棕色眼镜。嗓门不大，却能镇住全场。我跟她说“发生了什么？”她冷漠得说：",
                        out="没什么，你别管了。"))

print("====风格对话====")

while(1):
    print("输入Q退出")
    prompt = input("我：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt,trun="”")
    print(response+"”")
