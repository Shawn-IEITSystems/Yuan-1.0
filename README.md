# Yuan-1.0
Yuan 1.0:  Large-Scale Pre-trained Language Model in
Zero-Shot and Few-Shot Learning

# Introduction
Recent work like GPT-3 has demonstrated excellent performance of Zero-Shot and Few-Shot learning on many natural language processing (NLP) tasks by scaling up model size, data size and the amount of compute. While training a model like GPT-3 requires huge amount of computing power that is a challenge to the researchers. In this work, we propose a method that incorporates large-scale distributed training performance into model architecture design. With this method, we trained Yuan 1.0, the current largest singleton language model with 246B parameters, which achieved excellent performance on thousands GPUs, and state-of-the-art results on different natural language processing tasks. We propose a data processing method that can efficiently filter massive amount of data from Internet. A new dataset with 5TB high-quality text, the current largest Chinese text corpus, is built based on this method. We propose a method based on calibration and label expansion to improve the Zero-Shot and Few-Shot performance, and steady performance improvements were observed. The articles that Yuan 1.0 generated are difficult to distinguish from articles written by humans.


Please find details in the paper of Yuan-1.0.
https://arxiv.org/abs/2110.04725

## 1. Open source of Yuan-1.0

We will open the corpus (1TB) and API of the Yuan model, as well as the codes for fine-tune, few-shot and zero-shot learning. 
Please visit [official website](http://air.inspur.com) (https://air.inspur.com/home) for details to get access of the corpus and APIs of Yuan model.

## 2. Requirements
The inference code is provided on python3. Before start using Yuan API to build your application, several python libs are required. You can simply install them via pip tools.
``` bash
pip install requests hashlib json
```
## 3. How to use Yuan-API
After submit application on official website, it will take several days (normally less than one week) for us to check your application. 

Please keep your registered account and phone number properly, which will be used to generate an unique key to get access the API.

For more details, please check the example code, `API/yuan_api_demo.py`, and follow the API document.
