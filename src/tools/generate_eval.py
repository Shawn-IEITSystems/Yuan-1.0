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
from sklearn.metrics import confusion_matrix


def eval_accuracy(all_label_probs, test_labels, labels_dict_sorted, mode=None, p_cf=None, f_log=None):
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

    correctness_list = []
    #print(len(all_label_probs))
    #print(len(test_labels))
    assert len(all_label_probs) == len(test_labels)
    pred_labels = []
    for label_probs, true_label in zip(all_label_probs, test_labels):
        f_log.write('before normalize:{}'.format(label_probs)+ '\n')
        label_probs = label_probs / np.sum(label_probs) # normalize to 1
        f_log.write('after normalize to 1:{}'.format(label_probs)+ '\n')
        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        f_log.write('after calibrate label probs:{}'.format(calibrate_label_probs)+ '\n')
        
        ## 扩充label方案
        ## 以tnews为例，计算到此处calibrate_label_probs75维，即每个标签校正后结果
        ## 然后每5个数值取平均，再通过最大值获取对应标签值
        #calibrate_label_probs1 = calibrate_label_probs.reshape((len(labels_dict_sorted), -1))
        #calibrate_label_probs2 = np.average(calibrate_label_probs1, axis=1)  # 按行求均值
        #ans_label = np.argmax(calibrate_label_probs2)

        ans_label = np.argmax(calibrate_label_probs)
        f_log.write('np argmax:{}'.format(ans_label)+ '\n')
        ans_label = labels_dict_sorted[ans_label]
        f_log.write('pred label:{}'.format(ans_label)+ '\n')
        f_log.write('true label:{}'.format(true_label)+ '\n')
        pred_labels.append(ans_label)
        if str(ans_label) == str(true_label):
            correctness_list.append(1)
            print('same' + '\t' + str(true_label) + '\t' + str(ans_label))
            f_log.write('pred and true label is same'+ '\n')
        else:
            correctness_list.append(0)
            print('diff' + '\t' + str(true_label) + '\t' + str(ans_label))
            f_log.write('pred and true label is different'+ '\n')
    return np.mean(correctness_list), pred_labels


def get_probs_labels(input_params):
    task_type = input_params.task_type
    path = input_params.data_path
    output_path = input_params.output_path
    #output_results_path = os.path.join(output_path, task_type)
    output_results_path = output_path
    os.makedirs(output_results_path, exist_ok=True)
    f_log = open(os.path.join(output_results_path, 'log.txt'), 'w', encoding='utf-8')
    # with open('./result.txt', 'a+', encoding='utf-8') as f:
    #     f.write('localtime: {}'.format(time.asctime(time.localtime(time.time()))) + '\n')

    test_labels_path = os.path.join(path, task_type, 'test_labels.txt')
    # test_labels_path = args.test_labels_file
    with open(test_labels_path, 'r', encoding='utf-8') as f_labels:
        labels_str = f_labels.readlines()
        test_labels = [label.strip('\n') for label in labels_str]
        # print(len(test_labels))
    f_log.write('test_labels:{}'.format(test_labels) + '\n')

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
    acc_original, pred_labels_org = eval_accuracy(all_label_probs, test_labels, labels_dict_sorted, f_log=f_log)

    logits_free_path = os.path.join(output_results_path, 'output_logits_pcf.txt')
    #logits_free_path = os.path.join(output_results_path, 'output_logits.txt')
    if os.path.exists(logits_free_path):
        p_cf = get_p_content_free(logits_free_path)
    else:
        p_cf = None
    acc_calibrated, pred_labels_cal = eval_accuracy(all_label_probs, test_labels, labels_dict_sorted, mode="diagonal_W", p_cf=p_cf, f_log=f_log)
    accuracies = [acc_original, acc_calibrated]
    print(f"Accuracies: {accuracies}")
    print(f"p_cf      : {p_cf}")

    f_log.close()
    with open(os.path.join(output_results_path, 'result_acc.txt'), 'w', encoding='utf-8') as f:
        f.write("task: {}, original acc and calibrated acc: {}".format('cla', accuracies) + '\n')
        f.write("p_cf: {}".format(p_cf) + '\n')

    with open(os.path.join(output_results_path, 'result_preds.txt'), 'w', encoding='utf-8') as f_preds:
        f_preds.write('true_label' + '\t' +'pred_label_org' + '\t' + 'perd_label_cal' + '\n')
        for true_label, label_org, label_cal in zip(test_labels, pred_labels_org, pred_labels_cal):
            f_preds.write(str(true_label) + '\t' +str(label_org) + '\t' + str(label_cal) + '\n')

        cm_org = confusion_matrix(test_labels, pred_labels_org, labels=labels_dict_sorted)
        cm_cal = confusion_matrix(test_labels, pred_labels_cal, labels=labels_dict_sorted)

        print("confusion matrix before calibration")
        print(cm_org)
        print("confusion matrix after calibration")
        print(cm_cal)
        labels_dict_inv = [value[0] for key, value in labels_dict.items()]
        f_preds.write("\nconfusion matrix before calibration\n")
        for label in labels_dict_inv:
            f_preds.write('\t' + label)
        f_preds.write('\n')
        for i in range(len(cm_org)):
            f_preds.write(labels_dict_inv[i] + '\t')
            for j in range(len(cm_org)):
                f_preds.write(str(cm_org[i][j]) + '\t')
            f_preds.write('\n')

        f_preds.write("\nconfusion matrix after calibration\n")
        for label in labels_dict_inv:
            f_preds.write('\t' + label)
        f_preds.write('\n')
        for i in range(len(cm_cal)):
            f_preds.write(labels_dict_inv[i] + '\t')
            for j in range(len(cm_cal)):
                f_preds.write(str(cm_cal[i][j]) + '\t')
            f_preds.write('\n')


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
    p_y = p_y / np.sum(p_y) # normalize
    return p_y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--data_path', default='./tasks/fewclue_text', type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--task_type', default='tnews', type=str)
    input_params = parser.parse_args()

    get_probs_labels(input_params)
    pass
