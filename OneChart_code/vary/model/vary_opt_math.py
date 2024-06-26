#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from vary.utils.constants import *

from vary.model.plug.blip_process import BlipImageEvalProcessor

from vary.model.vision_encoder.sam import build_sam_vit_b


from transformers import OPTConfig, OPTModel, OPTForCausalLM

from vary.model.plug.transforms import train_transform, test_transform

from dataclasses import dataclass
import time 


class varyConfig(OPTConfig):
    model_type = "vary"


class varyOPTModel(OPTModel):
    config_class = varyConfig

    def __init__(self, config: OPTConfig):
        super(varyOPTModel, self).__init__(config)
        self.vision_tower = build_sam_vit_b()
        self.mm_projector = nn.Linear(1024, 768)


    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=True,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):

        # 224*224
        image_processor = None # CLIPImageProcessor.from_pretrained() 
        # 1024*1024
        image_processor_high = test_transform

        self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)


        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.use_im_start_end = use_im_start_end
        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )

    def embed_tokens(self, x):
        return self.get_input_embeddings()(x)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        vision_tower = getattr(self, 'vision_tower', None)
        
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)
            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)


            image_features = []
            for image in images:
                with torch.set_grad_enabled(True):
                    cnn_feature = vision_tower(image[1])
                    cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)
                    image_feature_final = cnn_feature
                image_features.append(image_feature_final)

            if type(images) is list:
                image_features = [self.mm_projector(image_feature) for image_feature in image_features]
            else:
                # image_features = self.mm_projector(image_features)
                raise NotImplementedError

            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            dummy_image_features = self.mm_projector(dummy_image_features)

            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]
                        # print(cur_input_ids)
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(varyOPTModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class varyOPTForCausalLM(OPTForCausalLM):
    config_class = varyConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        # print(config)
        super(OPTForCausalLM, self).__init__(config)
        self.model = varyOPTModel(config)

        self.num_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 256),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pred_locs = []

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, varyQwenModel):
    #         module.gradient_checkpointing = value

    def min_max_normalize_with_nan(self, tensor):
        # 排除NaN值的最小和最大值计算
        min_vals = torch.where(torch.isnan(tensor), torch.tensor(float('inf'), device=tensor.device), tensor).min(dim=1, keepdim=True).values
        max_vals = torch.where(torch.isnan(tensor), torch.tensor(float('-inf'), device=tensor.device), tensor).max(dim=1, keepdim=True).values
        
        # 应用归一化，同时保留NaN值不变
        normalized_tensor = torch.where(torch.isnan(tensor), tensor, (tensor - min_vals) / (max_vals - min_vals))
        return normalized_tensor


    def number_loss(self, src_numbers, target_numbers):
        target_numbers = torch.stack(target_numbers, dim=0)
        target_numbers = self.min_max_normalize_with_nan(target_numbers)
        mask = ~torch.isnan(target_numbers)
        loss_number = F.l1_loss(src_numbers[mask], target_numbers[mask], reduction='mean')
        return loss_number
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        loc_labels=None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        config = self.get_model().config

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        # 取出用于感知任务的 query hidden state
        if (loc_labels is not None) and len(loc_labels) > 0:
            det_patch_token = torch.where(input_ids == config.number_token)[1][0]
            pred_locs = self.num_decoder(hidden_states[:, det_patch_token, :]) # shape: [batch_size, 256]
        
        # inference时输出num_head预测的值
        if not self.training:
            try:
                det_patch_token = torch.where(input_ids == config.number_token)[1][0]
                pred_locs = self.num_decoder(hidden_states[:, det_patch_token, :]) # shape: [batch_size, 256]
                self.pred_locs = pred_locs[0][:100].cpu().tolist()
            except:
                pass


        # logits
        logits = self.lm_head(hidden_states).contiguous()
        loss = None
        lm_loss = None
        numbers_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = lm_loss

        # for 判别式 perception task
        if (loc_labels is not None) and len(loc_labels) > 0:
            numbers_loss = self.number_loss(pred_locs, loc_labels)
            loss = loss + numbers_loss
            print("lm_loss: ", lm_loss.cpu().item(), "numbers_loss: ", numbers_loss.cpu().item())


        if self.training:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
                "loc_labels": kwargs.get("loc_labels", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().config

        # add image patch token <image>
        tokenizer.add_tokens("</s>", special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        tokenizer.add_tokens(DEFAULT_IMAGE_PATCH_TOKEN, special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        config.im_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)


        # add image start token <im_start> and end token <im_end>
        if config.use_im_start_end:
            num_new_tokens = 2
            tokenizer.add_tokens(DEFAULT_IM_START_TOKEN , special_tokens=True)
            tokenizer.add_tokens(DEFAULT_IM_END_TOKEN , special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            config.im_start_token = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            config.im_end_token =  tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

            # config.im_start_token, config.im_end_token = 151857, 151858

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                
        tokenizer.add_tokens(DEFAULT_NUMBER_TOKEN, special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        config.number_token = tokenizer.convert_tokens_to_ids(DEFAULT_NUMBER_TOKEN)


AutoConfig.register("vary", varyConfig)
AutoModelForCausalLM.register(varyConfig, varyOPTForCausalLM)
