import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型
yuan = Yuan(engine='dialog',
            input_prefix="问：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            append_output_prefix_to_query=True)

# 3. add examples if in need.
yuan.add_example(Example(inp="对百雅轩798艺术中心有了解吗？",
                        out="有些了解，它位于北京798艺术区，创办于2003年。"))
yuan.add_example(Example(inp="不过去这里我不知道需不需要门票？",out="我知道，不需要，是免费开放。"))
yuan.add_example(Example(inp="你还可以到它边上的观复博物馆看看，我觉得那里很不错。",out="观复博物馆我知道，是马未都先生创办的新中国第一家私立博物馆。"))

print("====多轮对话机器人====")

h_dialog = []   # 存放历史对话：元素为ex

def get_relative_qa(prompt, h_dialog):
    """
    可以添加相关性计算，这里简单使用最近的一次对话
    """
    if not h_dialog:
        return []
    else:
        return [h_dialog[-1]]

def update_example(yuan,exs):
    ex_ids = []
    for ex in exs:
        ex_ids.append(ex.get_id())
        yuan.add_example(ex)
    return yuan, ex_ids

while 1:
    print("输入Q退出")
    prompt = input("问：")
    if prompt.lower() == "q":
        break
    if prompt[-1] == "”":
        prompt = prompt[:-1]
    exs = get_relative_qa(prompt, h_dialog)
    yuan, ex_ids = update_example(yuan, exs)
    response = yuan.submit_API(prompt=prompt,trun="”")
    if len(h_dialog)<10:    # 设置保存最多不超过10轮最近的历史对话
        h_dialog.append(Example(inp=prompt,out=response))
    else:
        del(h_dialog[0])
        h_dialog.append(Example(inp=prompt,out=response))
    for ex_id in ex_ids:
        yuan.delete_example(ex_id)
    print(response)
