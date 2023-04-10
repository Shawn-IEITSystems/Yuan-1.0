import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from yuan_api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
# set_yuan_account("账号", "手机号")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog','rhythm_poems']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型，'rhythm_poems'是古文模型
yuan = Yuan(engine='base_10B',
            max_tokens=200,
            input_prefix="",
            input_suffix="",
            output_prefix="",
            output_suffix="",
            append_output_prefix_to_query=False,
            topK=5,
            temperature=1,
            topP=0.8,
            frequencyPenalty=1.2)

# 3. add examples if in need.
yuan.add_example(Example('以“中秋”为题写一篇作文：',out='农历八月十五，是个团圆的日子。中午，我和爸爸妈妈去看望爷爷奶奶，奶奶突然谈论起在外地工作的姑姑，还说她到春节就要回来了。我心有感触，真是“每逢佳节倍思亲”哪。'))
print("====文章续写====")

while(1):
    print("输入Q退出")
    prompt = input("输入：")
    if prompt.lower() == "q":
        break
    response = yuan.submit_API(prompt=prompt)
    print(response+"")
