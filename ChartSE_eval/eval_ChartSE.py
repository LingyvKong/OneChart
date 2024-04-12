# Evaluation script for structural extraction task in ChartX
# Reference https://arxiv.org/abs/2309.11268 and 
# Written by Renqiu Xia, Hancheng Ye
# All Rights Reserved 2024-2025.

import os
import argparse
import json
import time
import logging
import datetime
from tqdm import tqdm
from SCRM import csv_eval, draw_SCRM_table,ocr_eval


# Load pred answer
# pred_type = "csv"
pred_type = "json"
with open('ChartSE_Benchmark/pred_results/onechart_chartqaSE.json', 'r') as questions_file:
    questions_data = json.load(questions_file)


# Load gt json
with open('ChartSE_Benchmark/ChartQA_test_human_filter.json', 'r') as values_file:
    values_data = json.load(values_file)


# 创建一个字典，将 "images" 映射到 "values"
image_to_values = {entry["images"]: entry["gts"]["values"] for entry in values_data}
title_to_values = {entry["images"]: entry["gts"]["title"] for entry in values_data}
source_to_values = {entry["images"]: entry["gts"]["source"] for entry in values_data}
x_title_to_values = {entry["images"]: entry["gts"]["x_title"] for entry in values_data}
y_title_to_values = {entry["images"]: entry["gts"]["y_title"] for entry in values_data}


def complete_json_string(json_str):
    """
    Attempt to complete a JSON string by ensuring it has matching opening and closing braces.
    
    Args:
    json_str (str): The input JSON string that might be incomplete.
    
    Returns:
    str: A possibly completed JSON string.
    """
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # 如果开括号多于闭括号，尝试补全闭括号
    while open_braces > close_braces:
        json_str += '}'
        close_braces += 1

    return json_str



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result_dir", required=False, help="Path to the inference result data")
    args = parser.parse_args()
    infer_result = args.infer_result_dir

    results = {}
    len_sum = 0

    csv_gt_set = []
    csv_pred_set = []

    title_gt_set = []
    title_pred_set = []

    source_gt_set = []
    source_pred_set = []

    x_title_gt_set = []
    x_title_pred_set = []

    y_title_gt_set = []
    y_title_pred_set = []


    for item in tqdm(questions_data):
        imgname = item["imagename"]
        if imgname in title_to_values.keys():
            if pred_type == "json":
                try:
                    complete_json_answer = json.loads(complete_json_string(item["answer"]))
                    title_pred = complete_json_answer['title']
                    title_gt = title_to_values[imgname]
                    source_pred = complete_json_answer['source']
                    source_gt = source_to_values[imgname]
                    x_title_pred = complete_json_answer['x_title']
                    x_title_gt = x_title_to_values[imgname]
                    y_title_pred = (complete_json_answer)['y_title']
                    y_title_gt = y_title_to_values[imgname]
                    tmp = (complete_json_answer)
                    if "values" in tmp.keys():
                        csv_pred = tmp['values']
                        csv_gt = image_to_values[imgname]
                    elif "data" in tmp.keys():
                        csv_pred = tmp['data']
                        csv_gt = image_to_values[imgname]
                    else:
                        csv_pred = (complete_json_answer)['values']
                except Exception as e:
                    print(e)
                    continue
            else:
                complete_json_answer = item["answer"]
                csv_pred = complete_json_answer['values'].replace('<0x0A>', '\\n')
                csv_pred = csv_pred.replace('|', '\\t')
                title_pred = complete_json_answer['title']
                source_pred = complete_json_answer['source']
                x_title_pred = complete_json_answer['x_title']
                y_title_pred = complete_json_answer['y_title']

                csv_gt = image_to_values[imgname]
                title_gt = title_to_values[imgname]
                source_gt = source_to_values[imgname]
                x_title_gt = x_title_to_values[imgname]
                y_title_gt = y_title_to_values[imgname]

            # print('-'*60)
            csv_gt_set.append(csv_gt)
            csv_pred_set.append(csv_pred)
            
            title_gt_set.append(title_gt)
            title_pred_set.append(title_pred)

            source_gt_set.append(source_gt)
            source_pred_set.append(source_pred)

            x_title_gt_set.append(x_title_gt)
            x_title_pred_set.append(x_title_pred)

            y_title_gt_set.append(y_title_gt)
            y_title_pred_set.append(y_title_pred)

    easy = 1
    len_sum = len_sum + len(csv_pred_set)

    em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high = csv_eval(csv_pred_set, csv_gt_set, easy, pred_type=pred_type)
    print('\n')
    title_ocr_socre = ocr_eval(title_gt_set, title_pred_set)
    source_ocr_socre = ocr_eval(source_gt_set, source_pred_set)
    x_title_ocr_socre = ocr_eval(x_title_gt_set, x_title_pred_set)
    y_title_ocr_socre = ocr_eval(y_title_gt_set, y_title_pred_set)
    print(y_title_ocr_socre)

    
    #输出结构正确率
    structure_accuracy = len(csv_pred_set)/len(values_data)

    result = {'s':{"value": [em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high], "len":len(csv_gt_set)}}
    results.update(result)

    result_table = draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high,title_ocr_socre,source_ocr_socre,x_title_ocr_socre,y_title_ocr_socre,structure_accuracy)

    logging.info('*************** Performance *****************')
    logging.info('\n'+ result_table)

    print(result_table)
    print(structure_accuracy * map_strict)
    print(structure_accuracy * map_slight)
    print(structure_accuracy * map_high)
