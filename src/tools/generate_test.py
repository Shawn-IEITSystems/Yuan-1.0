# -*- coding: utf-8  -*-
# @Time    : 2021/8/9 15:10
# @File    : test_eval.py

import os
import numpy as np
import json
import time
import argparse
import sys
# sys.path.append('../')
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
import matplotlib.pyplot as plt  # 绘图库
import itertools


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def tran_prob_label(all_label_probs, labels_dict_sorted, mode=None, p_cf=None, f_log=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
    f_log.write('num_classes:{}'.format(num_classes)+ '\n')
    f_log.write('p_cf_w:{}'.format(W)+ '\n')
    f_log.write('p_cf_b:{}'.format(b)+ '\n')

    pred_labels = []
    for label_probs in all_label_probs:
        f_log.write('before normalize:{}'.format(label_probs)+ '\n')
        label_probs = label_probs / np.sum(label_probs) # normalize to 1
        f_log.write('after normalize to 1:{}'.format(label_probs)+ '\n')
        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        f_log.write('after calibrate label probs:{}'.format(calibrate_label_probs)+ '\n')

        # 扩充label方案
        # 以tnews为例，计算到此处calibrate_label_probs75维，即每个标签校正后结果
        # 然后每5个数值取平均，再通过最大值获取对应标签值
        calibrate_label_probs1 = calibrate_label_probs.reshape((len(labels_dict_sorted), -1))
        calibrate_label_probs2 = np.average(calibrate_label_probs1, axis=1)  # 按行求均值
        ans_label = np.argmax(calibrate_label_probs2)

        # ans_label = np.argmax(calibrate_label_probs)
        f_log.write('np argmax:{}'.format(ans_label)+ '\n')
        ans_label = labels_dict_sorted[ans_label]
        f_log.write('pred label:{}'.format(ans_label)+ '\n')
        pred_labels.append(ans_label)

    return  pred_labels


def get_probs_labels(input_params):
    task_type = input_params.task_type
    path = input_params.data_path
    output_path = input_params.output_path
    #output_results_path = os.path.join(output_path, task_type)
    output_results_path = output_path
    os.makedirs(output_results_path, exist_ok=True)
    f_log = open(os.path.join(output_results_path, 'log.txt'), 'w', encoding='utf-8')

    # test_labels_path = os.path.join(path, task_type, 'test_labels.txt')
    # test_labels_path = args.test_labels_file
    # with open(test_labels_path, 'r', encoding='utf-8') as f_labels:
    #     labels_str = f_labels.readlines()
    #     test_labels = [label.strip('\n') for label in labels_str]
        # print(len(test_labels))
    # f_log.write('test_labels:{}'.format(test_labels) + '\n')

    labels_dict_path = os.path.join(path, task_type, 'labels.json')
    # labels_dict_path = args.sample_class_file
    with open(labels_dict_path, 'r', encoding='utf-8') as f_labels_dict:
        # labels_dict_str = f_labels_dict.read().replace("'", "\"")
        labels_dict = json.load(f_labels_dict)
        #print(labels_dict)
    f_log.write('labels_dict:{}'.format(labels_dict) + '\n')

    labels_dict_sorted = list(labels_dict.keys())
    #print(labels_dict_sorted)
    f_log.write('labels_dict_sorted:{}'.format(labels_dict_sorted) + '\n')

    # labels_dict_inv = {value[0]: key for key, value in labels_dict.items()}
    # print ("按值(value)排序:")
    # print(sorted(labels_dict.items(), key = lambda kv:(kv[1], kv[0])))

    probs_path = os.path.join(output_results_path, 'output_logits.txt')

    all_label_probs = []
    with open(probs_path, 'r', encoding='utf-8') as f_labels:
        probs_str = f_labels.readlines()
        # prob_list = []
        for prob_str in probs_str:
            if "Logits:" not in prob_str:
                continue
            else:
                probs = json.loads(prob_str.strip('\n').split("Logits:")[-1].replace("'", "\""))
                # print(probs)
                prob_list=probs[1:]
            all_label_probs.append(prob_list)

    all_label_probs = np.array(all_label_probs)
    pred_labels_org = tran_prob_label(all_label_probs, labels_dict_sorted, f_log=f_log)

    # logits_free_path = os.path.join(output_results_path, 'output_logits.txt')
    # p_cf = get_p_content_free_1(logits_free_path, path, task_type)
    logits_free_path = os.path.join(output_results_path, 'output_logits_pcf.txt')
    if os.path.exists(logits_free_path):
        p_cf = get_p_content_free(logits_free_path)
    else:
        p_cf = None
    pred_labels_cal = tran_prob_label(all_label_probs, labels_dict_sorted, mode="diagonal_W", p_cf=p_cf, f_log=f_log)
    # print(f"p_cf      : {p_cf}")

    # 在此处添加Json文件输出，按zero榜单要求
    # generate_predict_answers(pred_labels_org)  #校正之前
    test_data_path = os.path.join(path, task_type, 'test.json')
    out_path = output_results_path
    save_test_result(task_type, test_data_path, pred_labels_org, out_path=out_path, cal=False)
    save_test_result(task_type, test_data_path, pred_labels_cal, out_path=out_path, cal=True)
    f_log.close()



def get_p_content_free(logits_free_path):
    all_p_y = []
    with open(logits_free_path, 'r', encoding='utf-8') as f_labels:
        probs_str = f_labels.readlines()
        for prob_str in probs_str:
            if "Logits:" not in prob_str:
                continue
            else:
                probs = json.loads(prob_str.strip('\n').split("Logits:")[-1].replace("'", "\""))
                prob_list = probs[1:]
            all_p_y.append(prob_list)
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)# normalize
    return p_y


def get_p_content_free_1(logits_free_path, path, task_type):
    labels_dict_path = os.path.join(path, task_type, 'labels.json')
    # labels_dict_path = args.sample_class_file
    with open(labels_dict_path, 'r', encoding='utf-8') as f_labels_dict:
        labels_dict = json.load(f_labels_dict)
    labels_dict_sorted = list(labels_dict.keys())

    labels_path = os.path.join(path, task_type, 'test_labels.txt')
    # test_labels_path = args.test_labels_file
    with open(labels_path, 'r', encoding='utf-8') as f_labels:
        labels_str = f_labels.readlines()
        labels = [label.strip('\n') for label in labels_str]

    tran_labels = []
    for label in labels:
        for i in range(len(labels_dict_sorted)):
            if label == labels_dict_sorted[i]:
                tran_labels.append(i)
                continue

    all_p_y = []
    with open(logits_free_path, 'r', encoding='utf-8') as f_labels:
        probs_str = f_labels.readlines()
        for prob_str in probs_str:
            if "Logits:" not in prob_str:
                continue
            else:
                probs = json.loads(prob_str.strip('\n').split("Logits:")[-1].replace("'", "\""))
                prob_list = probs[1:]
            all_p_y.append(prob_list)

    for i in range(len(tran_labels)):
        for j in range(len(all_p_y[i])):
            if j != tran_labels[i]:
                all_p_y[i][j] = 0

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) + 1e-10 # normalize
    return p_y


def save_test_result(task, test_data_path, pred_labels, out_path='', cal=True):
    # task，任务类型，取值'eprstmt'，'wsc'，'ocnli'等
    # test_data_path test.json所在位置，如'/es01/home/test.json'
    # pred_labels 预测标签结果列表
    # out_path 输出预测结果路径
    # for fname in test_data_path:
    # csldcp_predict: {"id": 0, "label": "力学"}{"id": 1, "label": "交通运输工程"}
    #cluewscf_predict:{"id": 49, "label": "false"}{"id": 50, "label": "false"}
    # chidf_predict:{"id": 3000, "answer": 0} {"id": 3001, "answer": 0}
    # bustm_predict:{"id": 3000, "label": "1"}{"id": 3001, "label": "0"}
    # tnewsf_predict:{"id": "0", "label": "101"}
    # ocnlif_predict:{"label":"neutral","id":0}
    # iflytekf_predict:{"id": 0, "label": "99"}
    # eprstmt_predict:{"id": 111, "label": "Negative"}
    # cslf_predict:{"id": 2415, "label": "0"}

    if task in ['tnews', 'ocnli', 'iflytek', 'csl', 'chid']:
        task_name = task + 'f'
    elif task == 'wsc':
        task_name = 'cluewscf'
    else:
        task_name = task
    if cal == True:
        out_name = os.path.join(out_path, task_name + '_predict_cal.json')
    else:
        out_name = os.path.join(out_path, task_name+'_predict.json')

    with open(out_name, 'w', encoding='utf-8') as fout:
        with open(test_data_path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == len(pred_labels), 'predict result length ({}) not eq test data length ({}) in  {}'.format(len(pred_labels), len(lines), test_data_path)
            for i, line in enumerate(lines):
                d = json.loads(line)
                r = {}
                r['id'] = d['id']

                if task == 'eprstmt':
                    if int(pred_labels[i]) == 0:
                        r['label'] = 'Negative'
                    else:
                        r['label'] = 'Positive'
                elif task == 'wsc':
                    if int(pred_labels[i]) == 0:
                        r['label'] = 'false'
                    else:
                        r['label'] = 'true'
                elif task == 'ocnli':
                    labels = ["neutral", "entailment", "contradiction"]
                    r['label'] = labels[int(pred_labels[i])]
                elif task == 'chid':
                    r['answer'] = pred_labels[i]
                else:
                    r['label'] = pred_labels[i]

                s = json.dumps(r, ensure_ascii=False)
                fout.write(s + '\n')



if __name__ == '__main__':

    # data_path 数据路径包含label.json test.json
    # output_path 输出路径，包含输出logits.txt，和输出test_predict.json
    # task_type 任务类型
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--data_path', default='/mnt/zhaoxd/yutong/metabrain-plm-lhl-new/Megatron-LM/tasks/fewclue_text', type=str)
    parser.add_argument('--output_path', default='/mnt/zhaoxd/yutong/metabrain-plm-lhl-new/Megatron-LM/tasks/fewclue_text/iflytek', type=str)
    parser.add_argument('--task_type', default='iflytek', type=str)
    input_params = parser.parse_args()

    get_probs_labels(input_params)
    pass
