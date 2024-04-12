# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import logging
import pathlib
import torch
import transformers

from vary.train.trainer_vit_fixlr import varyTrainer
from vary.model import varyLlamaForCausalLM
from vary.model import *
from vary.data import make_supervised_data_module
from vary.utils.arguments import *
from vary.utils.constants import *
from vary.utils.utils import smart_tokenizer_and_embedding_resize
from vary.model.vision_encoder.vitg_qwen import create_clip_vit_g, create_perciever
from vary.model.vision_encoder.sam import build_sam_vit_b

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, padding_side="right", model_max_length=training_args.model_max_length)

    model = varyOPTForCausalLM.from_pretrained(model_args.model_name_or_path)

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    vision_tower_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=model_args.freeze_vision_tower,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device
    )

    model.initialize_vision_tokenizer(
        tokenizer=tokenizer, 
        freeze_lm_model=model_args.freeze_lm_model, 
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )

    model.to(dtype=dtype, device=training_args.device)
    data_args.image_token_len = 256
    data_args.image_processor = vision_tower_dict['image_processor']
    data_args.image_processor_high = vision_tower_dict['image_processor_high']
    data_args.use_im_start_end = model_args.use_im_start_end

    # mixed relation, to be fixed
    if model_args.freeze_lm_model:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True

        if not model_args.freeze_vision_tower:
            model.get_model().vision_tower.requires_grad_(True)


    if model_args.freeze_vision_tower:
        model.get_model().vision_tower.requires_grad_(False)
                
    params_grad = [p.numel() for n, p in model.named_parameters() if p.requires_grad]
    print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

    data_module = make_supervised_data_module(
        interleave=training_args.interleave, 
        with_box=training_args.with_box, 
        tokenizer=tokenizer, 
        data_args=data_args
    )

    trainer = varyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._safe_save(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()



# deepspeed /data/OneChart_code/vary/train/train_opt.py     --deepspeed /data/OneChart_code/zero_config/zero2.json --model_name_or_path /data/checkpoints/varytiny/  --vision_tower /data/checkpoints/varytiny/ --freeze_vision_tower False --freeze_lm_model False --vision_select_layer -2 --use_im_start_end True --bf16 True --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 250 --save_total_limit 1 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --report_to none --per_device_train_batch_size 16 --num_train_epochs 1 --learning_rate 5e-5 --datasets render_chart_en+render_chart_zh  --output_dir /data/checkpoints/onechart-pretrain/