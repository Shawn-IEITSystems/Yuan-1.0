# Yuan 1.0 Sandbox: Turn your ideas into demos in a matter of minutes

Yuan 1.0 sandbox is developed on the basis of GPT-3 sandbox tool.

Initial release date: 12 March 2022

Note that this repository is not under any active development; just basic maintenance.

## Description

The goal of this project is to enable users to create cool web demos using the newly released Inspur Yuan 1.0 API **with just a few lines of Python.** 

This project addresses the following issues:

1. Automatically formatting a user's inputs and outputs so that the model can effectively pattern-match
2. Creating a web app for a user to deploy locally and showcase their idea

Here's a quick example of priming Yuan to build a chatbot:

```
# Construct Yuan object and show some examples
yuan = Yuan(input_prefix="对话：“",
            input_suffix="”",
            output_prefix="答：“",
            output_suffix="”",
            append_output_prefix_to_query=False)
yuan.add_example(Example(inp="对百雅轩798艺术中心有了解吗？",
                        out="有些了解，它位于北京798艺术区，创办于2003年。"))
yuan.add_example(Example(inp="不过去这里我不知道需不需要门票？",out="我知道，不需要，是免费开放。"))
yuan.add_example(Example(inp="你还可以到它边上的观复博物馆看看，我觉得那里很不错。",out="观复博物馆我知道，是马未都先生创办的新中国第一家私立博物馆。"))

# Define UI configuration
config = UIConfig(description="旅游问答机器人",
                  button_text="回答",
                  placeholder="故宫里有什么好玩的？",
                  show_example_form=True)

demo_web_app(yuan, config)
```

Running this code as a python script would automatically launch a web app for you to test new inputs and outputs with. There are already 3 example scripts in the `examples` directory.

You can also prime Yuan from the UI. for that, pass `show_example_form=True` to `UIConfig` along with other parameters.

Technical details: the backend is in Flask, and the frontend is in React. Note that this repository is currently not intended for production use.

## Background

Yuan 1.0 ([Shawn et al.](https://arxiv.org/abs/2110.04725)) is Inspur's latest chinese language model. In this work, we propose a method that incorporates large-scale distributed training performance into model architecture design. With this method, we trained Yuan 1.0, the current largest singleton language model with 246B parameters, which achieved excellent performance on thousands GPUs, and state-of-the-art results on different natural language processing tasks.

Please visit [official website](http://air.inspur.com) (http://air.inspur.com) for details to get access of the corpus and APIs of Yuan model.

## Requirements

Coding-wise, you only need Python. But for the app to run, you will need:

* API key from the Inspur Yuan1.0 API invite
* Python 3
* `yarn`

Instructions to install Python 3 are [here](https://realpython.com/installing-python/), and instructions to install `yarn` are [here](https://classic.yarnpkg.com/en/docs/install/#mac-stable).

## Setup

First, clone or fork this repository. Then to set up your virtual environment, do the following:

1. Create a virtual environment in the root directory: `python -m venv $ENV_NAME`
2. Activate the virtual environment: ` source $ENV_NAME/bin/activate` (for MacOS, Unix, or Linux users) or ` .\ENV_NAME\Scripts\activate` (for Windows users)
3. Install requirements: `pip install -r api/requirements.txt`
4. To add your secret key: create a python file for your demo, and set your account and phone number on the begining of the file: `set_yuan_account("account", "phone")`, where `$account` and `$phone` are registed on our official website and obtained the authorization. If you are unsure whether you have the access permission, navigate to the [official website](http://air.inspur.com) to check the state.
5. Run `yarn install` in the root directory

If you are a Windows user, to run the demos, you will need to modify the following line inside `api/demo_web_app.py`:
`subprocess.Popen(["yarn", "start"])` to `subprocess.Popen(["yarn", "start"], shell=True)`

To verify that your environment is set up properly, run one of the 3 scripts in the `examples` directory:
`python examples/run_dialog.py`

A new tab should pop up in your browser, and you should be able to interact with the UI! To stop this app, run ctrl-c or command-c in your terminal.


## Contributions

We actively encourage people to contribute by adding their own examples or even adding functionalities to the modules. Please make a pull request if you would like to add something, or create an issue if you have a question. We will update the contributors list on a regular basis.

Please *do not* leave your secret key in plaintext in your pull request!

## Authors

We thank the original authors of GPT-3 sandbox tool:

* Shreya Shankar
* Bora Uyumazturk
* Devin Stein
* Gulan
* Michael Lavelle

