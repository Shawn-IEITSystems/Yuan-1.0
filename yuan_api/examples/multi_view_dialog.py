import os
import sys
sys.path.append(os.path.abspath(os.curdir))
from simhash import Simhash
from yuan_api.inspurai import Yuan, set_yuan_account,Example
import heapq

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='dialog',
            input_prefix="问：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            max_tokens=30,
            append_output_prefix_to_query=True)

# 3. add examples if in need.

print("====多轮对话机器人====")

h_dialog = []   # 存放历史对话：元素为ex

def get_relative_qa(prompt, h_dialog, topN=2):
    """
    可以添加相关性计算，这里简单使用最近的一次对话。
    :topN: 需要返回的相关对话轮数。
    """
    def simhash(query, text,):
        """
        采用局部敏感的hash值表示语义。
        """
        q_simhash = Simhash(query)
        t_simhash = Simhash(text)
        max_hashbit = max(len(bin(q_simhash.value)), len(bin(t_simhash.value)))

        distance = q_simhash.distance(t_simhash)
        # print(distance)

        similar = 1 - distance / max_hashbit
        return similar
    
    h_num = len(h_dialog)
    sim_values = []
    tm_effs= []
    rel_effs = []
    gamma = 0.8 # time effect coefficient

    if not h_dialog:
        return []
    else:
        for indx, dialog in enumerate(h_dialog):
            text = '|'.join((dialog.input, dialog.output))
            sim_value = simhash(prompt, text)
            tm_eff = gamma ** ((h_num - indx)/h_num)
            rel_eff = sim_value * tm_eff
            sim_values.append(sim_value)
            tm_effs.append(tm_eff)
            rel_effs.append(rel_eff)
        
        top_idx = heapq.nlargest(topN, range(len(rel_effs)), rel_effs.__getitem__)
        mst_dialog = [h_dialog[idx] for idx in top_idx]
        mst_dialog.reverse()
        return mst_dialog

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
    response = yuan.submit_API(prompt=prompt,trun="")
    if len(h_dialog)<10:    # 设置保存最多不超过10轮最近的历史对话
        h_dialog.append(Example(inp=prompt,out=response))
    else:
        del(h_dialog[0])
        h_dialog.append(Example(inp=prompt,out=response))
    for ex_id in ex_ids:
        yuan.delete_example(ex_id)
    print(response)
