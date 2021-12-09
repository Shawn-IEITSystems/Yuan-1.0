#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:air.inspur.com
@file:yuan_api_demo.py
@time:2021/11/24
"""
import requests
import hashlib
import time
import json


def code_md5(str):
    code=str.encode("utf-8")
    m = hashlib.md5()
    m.update(code)
    result= m.hexdigest()
    return result

def rest_get(url, header,timeout, show_error=False):
    '''Call rest get method'''
    try:
        response = requests.get(url, headers=header,timeout=timeout, verify=False)
        return response
    except Exception as exception:
        if show_error:
            print(exception)
        return None



if __name__ == '__main__':
    # 1、使用md5加密获得token
    t=time.strftime("%Y-%m-%d", time.localtime())
    account= "inspur" # 替换为申请的账号名
    phone= "123456789012"    # 替换为申请帐号时填写的手机号
    token=code_md5(account+phone+t)
    print(token)
    headers = {'token': token}

    # 2、发起推理请求
    ques="上联：园中草木春无数；下联：湖上山林画不如。上联：好事流传千古；下联："
    # ques= "春风用意匀颜色，销得携觞与赋诗。秾丽最宜新著雨，娇饶全在欲开时。以清风为题作一首诗："
    temperature=0.9  # 采样temperature，用于模型生成多样性，默认值0.9
    topP=0.1  # Top P 采样，默认值0.1
    topK=1  # Top K 采样，默认值1
    tokensToGenerate=20 # 生成tokens数目，建议与输入的token的个数之和小于2048
    url="http://api-air.inspur.com/v1/interface/api/requestId?"
    url=url+"account={0}&data={1}&temperature={2}&topP={3}&topK={4}&tokensToGenerate={5}&type={6}".format(account,ques,temperature,topP,topK,tokensToGenerate,"api")
    print(url)
    response=rest_get(url,headers,30)
    response_text = json.loads(response.text)
    if response_text["flag"]:
        requestId = response_text["resData"]
    else:
        print(response_text)
        exit()
    print(response_text)

    # 3、查询推理结果
    url = "http://api-air.inspur.com/v1/interface/api/result?"
    url = url + "account={0}&requestId={1}".format(account, requestId)
    print(url)
    # 调用get请求
    while (1):
        response = rest_get(url, headers, 30)
        response_text = json.loads(response.text)
        if response_text["resData"] != None:
            break
        if response_text["flag"] == False:
            print(response_text)
            exit()
    print(response_text)
    pass