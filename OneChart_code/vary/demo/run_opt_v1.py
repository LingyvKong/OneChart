import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria

from PIL import Image
import json
import numpy as np
import re
import requests
from io import BytesIO
from transformers import TextStreamer
from vary.model.plug.transforms import test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'



def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def list_json_value(json_dict):
    rst_str = []
    sort_flag = True
    try:
        for key, value in json_dict.items():
            if isinstance(value, dict):
                decimal_out = list_json_value(value)
                rst_str = rst_str + decimal_out
                sort_flag = False
            elif isinstance(value, list):
                return []
            else:
                if isinstance(value, float) or isinstance(value, int):
                    rst_str.append(value)
                else:
                    # num_value = value.replace("%", "").replace("$", "").replace(" ", "").replace(",", "")
                    value = re.sub(r'\(\d+\)|\[\d+\]', '', value)
                    num_value = re.sub(r'[^\d.-]', '', str(value)) 
                    if num_value not in ["-", "*", "none", "None", ""]:
                        rst_str.append(float(num_value))
    except Exception as e:
        print(f"Error: {e}")
        # print(num_value)
        print(json_dict)
        return []
    # if len(rst_str) > 0:
    #     rst_str = rst_str + [float(-1)]
    return rst_str

def norm_(rst_list):
    if len(rst_list) < 2:
        return rst_list
    min_vals = min(rst_list)
    max_vals = max(rst_list)
    rst_list = np.array(rst_list)
    normalized_tensor = (rst_list - min_vals) / (max_vals - min_vals + 1e-9)
    return list(normalized_tensor)

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="right")


    model = varyOPTForCausalLM.from_pretrained(model_name)
    model.to(device='cuda',  dtype=torch.bfloat16)


    image_processor_high =  test_transform
    use_im_start_end = True

    image_token_len = 256

    chose_prompt = {
        '1': "Convert the key information of the chart to a python dict:",
        '2': "stop",
    }
    query = input('Query: [1] chart [2] stop: ')
    query = query.strip()
    if query in chose_prompt.keys():
        query = chose_prompt[query]
    while True:
        if query == 'stop':
            break
        print(query)
        image_file = input('[4] stop [5] new conv / Image file: ')
        image_file = image_file.strip()
        if image_file in ['4', 'stop']:
            break
        if image_file in ['5', 'new conv']:
            query = input('Query: [1] chart [2] stop: ')
            query = query.strip()
            if query in chose_prompt.keys():
                query = chose_prompt[query]
            if query == 'stop':
                break
            image_file = input('[5] new conv / Image file: ')
            image_file = image_file.strip()
        
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + query + '\n'


        conv_mode = "v1"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # prompt = prompt + "<Number>"
        inputs = tokenizer([prompt])


        try:
            image = load_image(image_file)
        except:
            print("imgpath is not exist, please check.")
            continue
        image_1 = image.copy()
        image_tensor_1 = image_processor_high(image_1).to(torch.bfloat16)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = '</s>'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        flag = 0

        with torch.autocast("cuda", dtype=torch.bfloat16): # bfloat16
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor_1.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).cuda())],
                do_sample=False,
                num_beams = 1,
                streamer=streamer,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria]
                )
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        outputs = outputs.replace("<Number> ", "")
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        if outputs[-1] == '.':
            outputs = outputs[:-1]
        
        pred_nums = model.pred_locs
        try:
            outputs_json = json.loads(outputs)
            list_v = list_json_value(outputs_json['values'])
            list_v = [round(x,4) for x in norm_(list_v)]
            gt_nums = torch.tensor(list_v).reshape(1,-1)
            print("<Chart>: ", pred_nums[:len(list_v)])
            pred_nums_ = torch.tensor(pred_nums[:len(list_v)]).reshape(1,-1)
            reliable_distence = F.l1_loss(pred_nums_, gt_nums)
            print("reliable_distence: ", reliable_distence)
            if reliable_distence < 0.1:
                print("After OneChart checking, this prediction is reliable.")
            else:
                print("This prediction may be has error! ")
        except Exception as e:
            print("This prediction may be has error! ")
            print(e)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default='v1')
    args = parser.parse_args()

    eval_model(args)
