import re
import string
from typing import Any, Callable, Optional, Sequence
import datasets
import numpy as np
import Levenshtein
import editdistance

def get_anls(s1, s2):
    try:
        s1 = s1.lower()
        s2 = s2.lower()
    except:
        pass
    if s1 == s2:
        return 1.0
    iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
    anls = iou
    return anls


def ocr_eval(references,predictions):
    socre_=0.0
    None_num=0
    for idx,ref_value in enumerate(references):
        pred_value = predictions[idx]
        pred_values, ref_values = [], []
        if isinstance(pred_value, str):
            pred_values.append(pred_value)
        else:
            pred_values = pred_value
        if isinstance(ref_value, str):
            ref_values.append(ref_value)
        else:
            ref_values = ref_value
        
        temp_score = 0.0
        temp_num = len(ref_values)
        
        
        for tmpidx, tmpref in enumerate(ref_values):
            tmppred = pred_values[tmpidx] if tmpidx < len(pred_values) else pred_values[0]
            if len(pred_values) == 1 and tmppred != "None" and "None" not in ref_values:  # pred 1, and not None
                temp_score = max(temp_score, get_anls(tmppred, tmpref))
                temp_num = len(ref_values)
            else:
                if tmppred=='None' and tmpref!='None':
                    temp_score += 0.0
                elif tmpref=='None':
                    temp_num -= 1
                else:
                    temp_score += get_anls(tmppred, tmpref)
        if temp_num == 0:
            ocr_score = 0.0
            None_num += 1
        else:
            ocr_score = temp_score / (temp_num)
        socre_ += ocr_score
    if None_num == len(references):
        return 9999
    else:
        return round(socre_ / (len(references)-None_num), 5)


def csv_eval(predictions,references,easy, pred_type='json'):
    predictions = predictions
    labels = references
    def is_int(val):
        try:
            int(val)
            return True
        except ValueError:
            return False

    def is_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False
    
    def convert_dict_to_list(data):
        """
        Convert a dictionary to a list of tuples, handling both simple and nested dictionaries.
        
        Args:
        data (dict): The input dictionary, which might be nested or simple.
        
        Returns:
        list: A list of tuples generated from the input dictionary.
        """
        # print(data)
        converted_list = []
        for key, value in data.items():
            # Check if the value is a dictionary (indicating a nested structure)
            if isinstance(value, dict):
                # Handle nested dictionary
                for subkey, subvalue in value.items():
                    # converted_list.append((key, subkey, subvalue))
                    converted_list.append((key, subkey, re.sub(r'[^\d.-]', '', str(subvalue))))

            else:
                # Handle simple key-value pair
                # converted_list.append((key, "value", value))
                converted_list.append((key, "value", re.sub(r'[^\d.-]', '', str(value))))
        return converted_list


    def csv2triples(csv, separator='\\t', delimiter='\\n'):  
        lines = csv.strip().split(delimiter)
        header = lines[0].split(separator) 
        triples = []
        for line in lines[1:]:   
            if not line:
                continue
            values = line.split(separator)
            entity = values[0]
            for i in range(1, len(values)):
                if i >= len(header):
                    break
                #---------------------------------------------------------
                temp = [entity.strip(), header[i].strip()]
                temp = [x if len(x)==0 or x[-1] != ':' else x[:-1] for x in temp]
                value = values[i].strip()
                value = re.sub(r'[^\d.-]', '', str(value))
                # value = value.replace("%","")     
                # value = value.replace("$","")     
                triples.append((temp[0], temp[1], value))
                #---------------------------------------------------------
        return triples
    
    def csv2triples_noheader(csv, separator='\\t', delimiter='\\n'):  
        lines = csv.strip().split(delimiter)
        maybe_header = [x.strip() for x in lines[0].split(separator)]
        not_header = False
        if len(maybe_header) > 2:
            for c in maybe_header[1:]:
                try:
                    num = float(c)
                    not_header = True
                except:
                    continue
                if not_header:
                    break
        header = None if not_header else maybe_header
        data_start = 0 if not_header and separator in lines[0] else 1
        triples = []
        for line in lines[data_start:]:   
            if not line:
                continue
            values = [x.strip() for x in line.split(separator)]
            entity = values[0]
            for i in range(1, len(values)):
                try:
                    temp = [entity if entity[-1]!=':' else entity[:-1], ""]
                except:
                    temp = [entity, ""]
                if header is not None:
                    try:
                        this_header = header[i]
                        temp = [entity, this_header]
                        temp = [x if x[-1] != ':' else x[:-1] for x in temp]
                    except:
                        this_header = entity.strip()
                value = values[i].strip()
                value = re.sub(r'[^\d.-]', '', str(value))
                # value = value.replace("%","")     
                # value = value.replace("$","")     
                triples.append((temp[0], temp[1], value))
                #---------------------------------------------------------
        return triples

    def process_triplets(triplets):
        new_triplets = []
        for triplet in triplets:
            new_triplet = []
            triplet_temp = []
            if len(triplet) > 2:
                if is_int(triplet[2]) or is_float(triplet[2]):
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), float(triplet[2]))
                else:
                    triplet_temp = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
            else: 
                triplet_temp = (triplet[0].lower(), triplet[1].lower(), "no meaning")
            new_triplets.append(triplet_temp)
        return new_triplets

    def intersection_with_tolerance(a, b, tol_word, tol_num):
        a = set(a)
        b = set(b)
        c = set()
        for elem1 in a:
            for elem2 in b:
                if is_float(elem1[-1]) and is_float(elem2[-1]):
                    if ((Levenshtein.distance(''.join(elem1[:-1]),''.join(elem2[:-1])) <= tol_word) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num))or \
                    ((''.join(elem1[:-1]) in ''.join(elem2[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)) or \
                    ((''.join(elem2[:-1]) in ''.join(elem1[:-1])) and (abs(elem1[-1] - elem2[-1]) / (abs(elem2[-1])+0.000001) <= tol_num)):
                        c.add(elem1)
                else:
                    if (Levenshtein.distance(''.join([str(i) for i in elem1]),''.join([str(j) for j in elem2])) <= tol_word):
                        c.add(elem1)
        return list(c)

    def union_with_tolerance(a, b, tol_word, tol_num):
        c = set(a) | set(b)
        d = set(a) & set(b)
        e = intersection_with_tolerance(a, b, tol_word, tol_num)
        f = set(e)
        g = c-(f-d)
        return list(g)

    def get_eval_list(pred_csv, label_csv, separator='\\t', delimiter='\\n', tol_word=3, tol_num=0.05, pred_type='json'):

        if pred_type == 'json':
            pred_triple_list=[]
            for it in pred_csv:
                pred_triple_temp = convert_dict_to_list(it)
                pred_triple_pre = process_triplets(pred_triple_temp)
                pred_triple_list.append(pred_triple_pre) 
        else:
            pred_triple_list=[]
            for it in pred_csv:
                pred_triple_temp = csv2triples(it, separator=separator, delimiter=delimiter)
                # pred_triple_temp = csv2triples_noheader(it, separator=separator, delimiter=delimiter)
                pred_triple_pre = process_triplets(pred_triple_temp)
                pred_triple_list.append(pred_triple_pre) 

        label_triple_list=[]
        for it in label_csv:
            label_triple_temp = convert_dict_to_list(it)
            label_triple_pre = process_triplets(label_triple_temp)
            label_triple_list.append(label_triple_pre) 

            
        intersection_list=[]
        union_list=[]
        sim_list=[]
        # for each chart image
        for pred,label in zip(pred_triple_list, label_triple_list):
            for idx in range(len(pred)):
                try:
                    if label[idx][1] == "value" and "value" not in pred[idx][:2]:
                        pred[idx] = (pred[idx][0], "value", pred[idx][2]) 
                    temp_pred_head = sorted(pred[idx][:2])
                    temp_gt_head = sorted(label[idx][:2])
                    pred[idx] = (temp_pred_head[0], temp_pred_head[1], pred[idx][2])
                    label[idx] = (temp_gt_head[0], temp_gt_head[1], label[idx][2])
                except:
                    continue
            intersection = intersection_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            union = union_with_tolerance(pred, label, tol_word = tol_word, tol_num=tol_num)
            sim = len(intersection)/len(union)
            intersection_list.append(intersection)
            union_list.append(union)
            sim_list.append(sim)
        return intersection_list, union_list, sim_list

    def get_ap(predictions, labels, sim_threhold, tolerance, separator='\\t', delimiter='\\n', easy=1):
        if tolerance == 'strict':
            tol_word=0
            if easy == 1:
                tol_num=0
            else:
                tol_num=0.1

        elif tolerance == 'slight':
            tol_word=2
            if easy == 1:
                tol_num=0.05
            else:
                tol_num=0.3

        elif tolerance == 'high':
            tol_word= 5
            if easy == 1:
                tol_num=0.1
            else:
                tol_num=0.5      
        intersection_list, union_list, sim_list = get_eval_list(predictions, labels, separator=separator, delimiter=delimiter, tol_word=tol_word, tol_num=tol_num, pred_type=pred_type)
        ap = len([num for num in sim_list if num >= sim_threhold])/(len(sim_list)+1e-16)
        return ap   

    map_strict = 0
    map_slight = 0
    map_high = 0
    s="\\t"
    d="\\n"

    for sim_threhold in np.arange (0.5, 1, 0.05):
        map_temp_strict = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='strict', separator=s, delimiter=d, easy=easy)
        map_temp_slight = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='slight', separator=s, delimiter=d, easy=easy)
        map_temp_high = get_ap(predictions, labels, sim_threhold=sim_threhold, tolerance='high', separator=s, delimiter=d, easy=easy)
        map_strict += map_temp_strict/10
        map_slight += map_temp_slight/10
        map_high += map_temp_high/10

    em = get_ap(predictions, labels, sim_threhold=1, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_strict = get_ap(predictions, labels, sim_threhold=0.5, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_75_strict = get_ap(predictions, labels, sim_threhold=0.75, tolerance='strict', separator=s, delimiter=d, easy=easy)    
    ap_90_strict = get_ap(predictions, labels, sim_threhold=0.90, tolerance='strict', separator=s, delimiter=d, easy=easy)
    ap_50_slight = get_ap(predictions, labels, sim_threhold=0.5, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_75_slight = get_ap(predictions, labels, sim_threhold=0.75, tolerance='slight', separator=s, delimiter=d, easy=easy)    
    ap_90_slight = get_ap(predictions, labels, sim_threhold=0.90, tolerance='slight', separator=s, delimiter=d, easy=easy)
    ap_50_high = get_ap(predictions, labels, sim_threhold=0.5, tolerance='high', separator=s, delimiter=d, easy=easy)
    ap_75_high = get_ap(predictions, labels, sim_threhold=0.75, tolerance='high', separator=s, delimiter=d, easy=easy)    
    ap_90_high = get_ap(predictions, labels, sim_threhold=0.90, tolerance='high', separator=s, delimiter=d, easy=easy)


    return em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high

def draw_SCRM_table(em, map_strict, map_slight, map_high, ap_50_strict, ap_75_strict, ap_90_strict, ap_50_slight, ap_75_slight, ap_90_slight, ap_50_high, ap_75_high, ap_90_high,title_ocr_socre,source_ocr_socre,x_title_ocr_socre,y_title_ocr_socre,structure_accuracy):

    result=f'''
            -----------------------------------------------------------\n
            |  Metrics   |  Sim_threshold  |  Tolerance  |    Value    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % map_strict}    |     \n
            |             |                 ----------------------------\n
            |  mPrecison  |  0.5:0.05:0.95  |   slight    |    {'%.4f' % map_slight}    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % map_high}    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_50_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |       0.5       |   slight    |    {'%.4f' % ap_50_slight }    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_50_high }    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_75_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |      0.75       |   slight    |    {'%.4f' % ap_75_slight}    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_75_high}    |\n
            -----------------------------------------------------------\n
            |             |                 |   strict    |    {'%.4f' % ap_90_strict}    |\n
            |             |                  ---------------------------\n
            |  Precison   |       0.9       |   slight    |    {'%.4f' % ap_90_slight }    |\n
            |             |                  ---------------------------\n
            |             |                 |    high     |    {'%.4f' % ap_90_high}    |\n
            -----------------------------------------------------------\n
            |Precison(EM) |                                    {'%.4f' % em}    |\n
            -----------------------------------------------------------\n
            |Title(EM)    |                                    {'%.4f' % title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |Source(EM)   |                                    {'%.4f' % source_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |X_title(EM)  |                                    {'%.4f' % x_title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |Y_title(EM)  |                                    {'%.4f' % y_title_ocr_socre}    |\n
            -----------------------------------------------------------\n
            |structure_acc|                                    {'%.4f' % structure_accuracy}    |\n
            -----------------------------------------------------------\n


            '''
    return result



