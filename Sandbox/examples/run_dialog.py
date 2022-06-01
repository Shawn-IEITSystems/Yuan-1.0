import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from api import UIConfig
from api import demo_web_app
from api.inspurai import Yuan, set_yuan_account, Example

# 1. set account
set_yuan_account("account", "phone Num.")  # 输入您申请的账号和手机号

# 2. initiate yuan api
# 注意：engine必需是['base_10B','translate','dialog']之一，'base_10B'是基础模型，'translate'是翻译模型，'dialog'是对话模型
yuan = Yuan(engine='dialog',
            input_prefix="问：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            append_output_prefix_to_query=True,
            frequencyPenalty=1.2)

# 3. add examples if in need.
yuan.add_example(Example(inp="对百雅轩798艺术中心有了解吗？",
                        out="有些了解，它位于北京798艺术区，创办于2003年。"))
# Define UI configuration
config = UIConfig(description="旅游问答机器人",
                  button_text="回答",
                  placeholder="故宫里有什么好玩的？",
                  show_example_form=True)

demo_web_app(yuan, config)
