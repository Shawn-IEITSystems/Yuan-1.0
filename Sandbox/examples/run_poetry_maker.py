import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from api import UIConfig
from api import demo_web_app
from api.inspurai import Yuan, set_yuan_account,Example

# 1. set account
set_yuan_account("account", "phone Num.")  # 输入您申请的账号和手机号

# 2. initiate yuan api
yuan = Yuan(input_prefix="以",
            input_suffix="为题作一首诗：",
            output_prefix="答：",
            output_suffix="”",
            append_output_prefix_to_query=False,
            max_tokens=40)

# 3. add examples if in need.
yuan.add_example(Example(inp="清风",
                        out="春风用意匀颜色，销得携觞与赋诗。秾丽最宜新著雨，娇饶全在欲开时。"))
yuan.add_example(Example(inp="送别",
                         out="渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。"))
yuan.add_example(Example(inp="新年",
                         out="欢乐过新年，烟花灿九天。金龙腾玉宇，六出好耘田。"))
yuan.add_example(Example(inp="喜悦",
                         out="昔日龌龊不足夸，今朝放荡思无涯。春风得意马蹄疾，一日看尽长安花。"))

# print("====作诗机器人====")
# Define UI configuration
config = UIConfig(description="作诗机器人",
                  button_text="作诗",
                  # placeholder="以何为题作诗：",
                  placeholder="田园",
                  show_example_form=True)

demo_web_app(yuan, config)


# while(1):
#     print("输入Q退出")
#     prompt = input("以何为题作诗：")
#     if prompt.lower() == "q":
#         break
#     response = yuan.submit_API(prompt=prompt,trun="”")
#     print(response+"”")
