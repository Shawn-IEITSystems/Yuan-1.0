# Yuan-1.0
Yuan 1.0:  Large-Scale Pre-trained Language Model in
Zero-Shot and Few-Shot Learning

# Introduction
Recent work like GPT-3 has demonstrated excellent performance of Zero-Shot and Few-Shot learning on many natural language processing (NLP) tasks by scaling up model size, data size and the amount of compute. While training a model like GPT-3 requires huge amount of computing power that is a challenge to the researchers. In this work, we propose a method that incorporates large-scale distributed training performance into model architecture design. With this method, we trained Yuan 1.0, the current largest singleton language model with 246B parameters, which achieved excellent performance on thousands GPUs, and state-of-the-art results on different natural language processing tasks. We propose a data processing method that can efficiently filter massive amount of data from Internet. A new dataset with 5TB high-quality text, the current largest Chinese text corpus, is built based on this method. We propose a method based on calibration and label expansion to improve the Zero-Shot and Few-Shot performance, and steady performance improvements were observed. The articles that Yuan 1.0 generated are difficult to distinguish from articles written by humans.


Please find details in the paper of Yuan-1.0.
https://arxiv.org/abs/2110.04725

## 1. Open source of Yuan-1.0

We will open the corpus (1TB) and API of the Yuan model, as well as the codes for fine-tune, few-shot and zero-shot learning. 
Please visit [official website](https://air.inspur.com/home) (https://air.inspur.com/home) for details to get access of the corpus and APIs of Yuan model.

## 2. Requirements
The inference code is provided on python3. Before start using Yuan API to build your application, several python libs are required. You can simply install them via pip tools.
``` bash
pip install requests hashlib json
```
## 3. How to use Yuan-API
After submit application on official website, it will take several days (normally less than one week) for us to check your application. 

Please keep your registered account and phone number properly, which will be used to generate an unique key to get access the API.

For more details, please check the example code, `yuan_api/examples`, and follow the API document.

## 4. Applications
Here we summarize some simple application example configuration methods for users' reference. The parameters not mentioned therein have adopted default values.

|ID|app|model|prompt template|input prefix|input suffix|output prefix|truncation character|example|few-shot|
|--:|:---:|:--:|:--------|:----|:--------|:--------|:--------|:--------|:---:|
|0|dialog generation|dialog|问：“`用户输入`”答：“|问：“|”|答：“|”|故宫有什么好玩的？|support|
|1|content continuation|base_10B|`用户输入`|无|无|无|默认|徐凤年刚走入京大校门，已经有学生会迎新的同学走到了他面前，|not recommended|
|2|poetry maker|base_10B|以“`用户输入`”为题作一首诗：“|以“|”为题作一首诗：“|无|”|清风| recommended|
|3|关键词抽取|base_10B|为以下正文提取关键词。正文：`用户输入`；关键词：|为以下正文提取关键词。正文：|；|关键词：|。|帮我写一首诗，描写春天到了，百花盛开。| support|
|4|ch-en translation|translate|将下列英文/中文翻译成中文/英文。英文/中文：`用户输入`中文/英文：“|将下列英文/中文翻译成中文/英文。英文/中文：|无|中文/英文：“|”|自然派的哲学家也被称为“苏格拉底之前的哲学家” 。|not recommended|

Please look forward to more applications.