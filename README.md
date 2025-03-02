<h3><a href="https://github.com/LingyvKong/OneChart/blob/main/OneChart_paper.pdf">OneChart: Purify the Chart Structural Extraction via One Auxiliary Token</a></h3>
<a href="http://arxiv.org/abs/2404.09987"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 
<a href="https://onechartt.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href='https://huggingface.co/kppkkp/OneChart/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href="https://zhuanlan.zhihu.com/p/692607557"><img src="https://img.shields.io/badge/zhihu-yellow"></a> 

Jinyue Chen*, Lingyu Kong*, [Haoran Wei](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Chenglong Liu, [Zheng Ge](https://joker316701882.github.io/), Liang Zhao, [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)
	


<p align="center">
<img src="assets/logo.png" style="width: 100px" align=center>
</p>

## Release
- [2024/9/16] 🔥 Support quickly trying the demo using [huggingface](https://huggingface.co/kppkkp/OneChart/blob/main/README.md).
- [2024/7/21] 🎉🎉🎉 OneChart is accepted by ACM'MM 2024 **Oral**! (3.97%)
- [2024/4/21] 🔥🔥🔥 We have released the **web demo** in [Project Page](https://onechartt.github.io/). Have fun!!
- [2024/4/15] 🔥 We have released the [code](https://github.com/LingyvKong/OneChart), [weights](https://huggingface.co/kppkkp/OneChart/tree/main) and the benchmark [data](https://drive.google.com/drive/folders/1YmOvxq0DfOA9YKoyCZDjpnTIkPNoyegQ?usp=sharing). 


## Contents
- [0. Quickly try the demo using hugginface](#0-quickly-try-the-demo-using-hugginface)
- [1. Benchmark Data and Evaluation Tool](#1-benchmark-data-and-evaluation-tool)
- [2. Install](#2-install)
- [3. Demo](#3-demo)
- [4. Train](#4-train)

<p align="center">
<img src="assets/append_all.png" style="width: 700px" align=center>
</p>

## 0. Quickly try the demo using hugginface
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('kppkkp/OneChart', trust_remote_code=True, use_fast=False, padding_side="right")
model = AutoModel.from_pretrained('kppkkp/OneChart', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda')
model = model.eval().cuda()

# input your test image
image_file = 'image.png'
res = model.chat(tokenizer, image_file, reliable_check=True)
print(res)
```

## 1. Benchmark Data and Evaluation Tool
- Download the ChartSE images and jsons [here](https://drive.google.com/drive/folders/1YmOvxq0DfOA9YKoyCZDjpnTIkPNoyegQ?usp=sharing). 
- Modify json path at the beginning of `ChartSE_eval/eval_ChartSE.py`. Then run eval script:
   
```shell
python ChartSE_eval/eval_ChartSE.py
```

## 2. Install
- Clone this repository and navigate to the code folder
```bash
git clone https://github.com/LingyvKong/OneChart.git
cd OneChart/OneChart_code/
```
- Install Package
```Shell
conda create -n onechart python=3.10 -y
conda activate onechart
pip install -e .
pip install -r requirements.txt
pip install ninja
```
- Download the OneChart weights [here](https://huggingface.co/kppkkp/OneChart/tree/main). 
  
## 3. Demo
```Shell
python vary/demo/run_opt_v1.py  --model-name  /onechart_weights_path/
```
Following the instruction, type `1` first, then type image path.

## 4. Train
- Prepare your dataset json, the format example is:
```json
[
 {
  "image": "000000.png",
  "conversations": [
   {
    "from": "human",
    "value": "<image>\nConvert the key information of the chart to a python dict:"
   },
   {
    "from": "gpt",
    "value": "<Number>{\"title\": \"Share of children who are wasted, 2010\", \"source\": \"None\", \"x_title\": \"None\", \"y_title\": \"None\", \"values\": {\"Haiti\": \"6.12%\", \"Libya\": \"5.32%\", \"Morocco\": \"5.11%\", \"Lebanon\": \"4.5%\", \"Colombia\": \"1.45%\"}}",
    "Numbers": [6.12, 5.32, 5.11, 4.5, 1.45]
   }
  ],
 },
 {
   ...
 }
]
```
In case you don't want to use and train the auxiliary head, comment out this line [`data_dict['loc_labels'] = self.extract_numbers(data["conversations"])`](https://github.com/LingyvKong/OneChart/blob/868942ace688231ba74e7ab3f1fe028d6c4776c6/OneChart_code/vary/data/conversation_dataset_v1_with_number.py#L214), and the json format can be:
```json
[
 {
  "image": "000000.png",
  "conversations": [
   {
    "from": "human",
    "value": "<image>\nConvert the key information of the chart to a python dict:"
   },
   {
    "from": "gpt",
    "value": "{\"title\": \"Share of children who are wasted, 2010\", \"source\": \"None\", \"x_title\": \"None\", \"y_title\": \"None\", \"values\": {\"Haiti\": \"6.12%\", \"Libya\": \"5.32%\", \"Morocco\": \"5.11%\", \"Lebanon\": \"4.5%\", \"Colombia\": \"1.45%\"}}"
   }
  ]
 },
 {
   ...
 }
]
```

- Fill in the data path to `OneChart/OneChart_code/vary/utils/constants.py`. Then a example script is:
```shell
deepspeed /data/OneChart_code/vary/train/train_opt.py     --deepspeed /data/OneChart_code/zero_config/zero2.json --model_name_or_path /data/checkpoints/varytiny/  --vision_tower /data/checkpoints/varytiny/ --freeze_vision_tower False --freeze_lm_model False --vision_select_layer -2 --use_im_start_end True --bf16 True --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 250 --save_total_limit 1 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --report_to none --per_device_train_batch_size 16 --num_train_epochs 1 --learning_rate 5e-5 --datasets render_chart_en+render_chart_zh  --output_dir /data/checkpoints/onechart-pretrain/
```
- You can pay attention to modifying these parameters according to your needs: `--model_name_or_path`, `freeze_vision_tower`, `--datasets`, `--output_dir` 


## Acknowledgement
- [Vary](https://github.com/Ucas-HaoranWei/Vary): the codebase and initial weights we built upon!

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. They are also restricted to use that follow the license agreement of Vary, Opt. 


## Citation
If you find our work useful in your research, please consider citing OneChart:
```bibtex
@inproceedings{chen2024onechart,
  title={Onechart: Purify the chart structural extraction via one auxiliary token},
  author={Chen, Jinyue and Kong, Lingyu and Wei, Haoran and Liu, Chenglong and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={147--155},
  year={2024}
}
```
