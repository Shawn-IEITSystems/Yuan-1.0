import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='rhythm_poems',
            input_prefix="",
            input_suffix="",
            output_prefix="",
            output_suffix="”",
            topK=5,
            topP=0.8,
            max_tokens=10,
            temperature=1,
            frequencyPenalty=1.2,
            append_output_prefix_to_query=False)

# 3. add examples if in need.
yuan.add_example(Example(inp="以清风为题作一首诗：“",
                        out="春风用意匀颜色，销得携觞与赋诗。秾丽最宜新著雨，娇饶全在欲开时。”"))


def hidden_poetry(prompt,yuan,theme):

    header = '以' + theme + '为题作一首诗：“'
    res = ''
    poem = ''
    head_chs = list(prompt)
    head_chs.reverse()
    for i in range(4):
        try:
            ch = head_chs.pop()
        except:
            ch = ''
        query = header + poem + ch
        for j in range(3):
            no_resp = True
            res = yuan.submit_API(query,trun='”')
            res = res.replace('。','，')
            res = res.split('，')
            if len(res[0]) >= 7-len(ch):
                no_resp = False
                break
        if no_resp:
            return "题目有些难，模型无输出"
        res = res[0][:7-len(ch)]
        if i%2:
            pun = '。'
        else:
            pun = '，'
        poem += ch + res + pun
    return poem


print("====藏头诗机器人====")
while(1):
    print("输入Q退出")
    prompt = input("以何为题作诗：")
    if prompt.lower() == "q":
        break
    theme = input('作诗主题：')
    # 藏头诗逻辑
    response = hidden_poetry(prompt,yuan, theme)
    
    # response = yuan.submit_API(prompt=prompt,trun="”")
    print('\n'+response)

