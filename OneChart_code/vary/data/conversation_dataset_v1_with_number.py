
import io
import os
import copy
import json
import logging
import torch
import torch.nn.functional as F
import transformers
import random

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from vary.data.base_dataset import BaseDataset
from vary.utils.constants import *
from vary.utils import conversation as conversation_lib

# from vary.utils.constants import DEFAULT_DET_PATCH_TOKEN

class ConversationDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, datasets, tokenizer, multimodal_cfg):
        super(ConversationDataset, self).__init__(datasets, tokenizer, multimodal_cfg)
        # v1 version format conversation
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
        logging.warning("Formatting inputs into conversation type: v1")
        logging.warning("Loading data...")

        list_data_dict = []
        list_image_path = []
        for name in datasets.split("+"):
            dataset = CONVERSATION_DATA[name]

            data_path = dataset['annotations']
            data = json.load(open(data_path, "r"))

            list_data_dict.extend(data)

            image_path = dataset['images']
            list_image_path.extend([image_path] * len(data))

            logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")

        self.list_data_dict = list_data_dict
        self.list_image_path = list_image_path
        self.im_patch_token, self.im_start_token, self.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    def multimodal_processor(self, sources):
        for source in sources:
            if self.multimodal_cfg['sep_image_conv_front']:
                assert DEFAULT_IMAGE_TOKEN in source[0]['value']
                source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                source[0]['value'] = DEFAULT_IMAGE_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
                if self.multimodal_cfg['use_im_start_end']:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources
    
    def pad_or_clip_tensor_to_fixed_length(self, tensor, fixed_length=256, padding_value=0):
        current_length = tensor.size(0)
        if current_length < fixed_length:
            padding_needed = fixed_length - current_length
            return F.pad(tensor, (0, padding_needed), "constant", padding_value)
        else:
            return tensor[:fixed_length]
        
    def extract_numbers(self, sources):
        # 暂时只支持单轮对话
        nums = []
        for source in sources:
            if source["from"].lower() == 'gpt' and "Numbers" in source.keys():
                nums = source["Numbers"]
        # if len(nums) != 0:
        nums = torch.tensor(nums)
        nums = self.pad_or_clip_tensor_to_fixed_length(nums, fixed_length=256, padding_value=torch.nan)
        return nums

    def token_processor(self, sources):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "assistant": conv.roles[1], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"].lower()] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"].lower()]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())


        # Tokenize conversations
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()



        # for idx, ii in enumerate(input_ids):
        #     if ii[-1] != torch.tensor(self.tokenizer.eos_token_id) and ii[-1] != torch.tensor(self.tokenizer.pad_token_id):
        #         input_ids[idx][-1] = torch.tensor(self.tokenizer.eos_token_id)
        #         targets[idx][-1] = torch.tensor(self.tokenizer.eos_token_id)


        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO


        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):


            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)

  
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX # keep bos
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)

                if len(parts) != 2:
                    break
                parts[0] += sep

                # print(parts)

                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
            # box_patch_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_DET_PATCH_TOKEN)
            # target[target==box_patch_id] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    # print(conversations)
                    # print(targets)
                    # exit()
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
        return dict(input_ids=input_ids, labels=targets)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # data = self.list_data_dict[i]
        data = copy.deepcopy(self.list_data_dict[i])
        if isinstance(data, dict):
            if 'image' in data:
                image_path = self.list_image_path[i]
                image_file = data['image']                
                image = Image.open(image_path + image_file).convert('RGB')

                try:
                    image, image_1 = self.image_processor(image)
                except:
                    print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
                    return self.__getitem__(0)

            conversations = self.multimodal_processor([data["conversations"]])
        else:
            conversations = [data]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_processor(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


        # check out if appended image token exceeds the maximum length
        images_left = torch.where(data_dict["input_ids"] == self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0])[0]
        num_images_right = 0
        if isinstance(data, dict) and ('image' in data or 'images' in data)and images_left.shape[0] > 0:
            images_right = images_left + self.multimodal_cfg['image_token_len'] + 1
            num_images_right = torch.where(images_right < data_dict["input_ids"].shape[0])[0].shape[0]
            if num_images_right < images_left.shape[0]:
                data_dict["input_ids"] = torch.cat([data_dict["input_ids"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])
                data_dict["labels"] = torch.cat([data_dict["labels"][:images_left[num_images_right]], torch.tensor([self.tokenizer.eos_token_id])])
        
        if isinstance(data, dict) and 'image' in data:
            data_dict['image'] = [image]
            data_dict['image_high'] = [image_1]
        else:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            data_dict['image_high'] = [torch.zeros(3, 1024, 1024)]
        data_dict['loc_labels'] = self.extract_numbers(data["conversations"])
        return data_dict

