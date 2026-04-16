# Mini Qwen

> 大模型训练全流程项目代码，by 古希腊掌管代码的神



## Python Environment

- 安装环境（除了grpo）

```bash
pip install -r requirements.txt
```

- 安装环境（grpo）

```bash
pip install -r requirements_grpo.txt
```



## Download Data

> 所以需要的数据都写在这个脚本里了，如果需要分次下载请进入脚本注释

```bash
bash download_data.sh
```



## Pretrain

- single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python mini_qwen_pt.py
```

- multi GPU

```bash
accelerate launch --config_file accelerate_config.yaml qwen_pt.py

# 后台运行，最好改成绝对路径
nohup accelerate launch --config_file accelerate_config.yaml qwen_pt.py > logs/output_pt.log 2>&1 &
```

> 混合精度训练: 全精度加载模型，并且traning_args.fp16 = True, 即使用 `--fp16` 参数
> 单精度训练: 全精度加载模型，并且traning_args.fp16 = False, 即不使用 `--fp16` 参数，也不使用 `--bf16` 参数



## XXX

这里的xxx代码python脚本，包括sft、distill等

- single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python xxx.py
```

- multi GPU

```bash
accelerate launch --config_file accelerate_config.yaml xxx.py

# 后台运行，最好改成绝对路径
nohup accelerate launch --config_file accelerate_config.yaml xxx.py > logs/output_pt.log 2>&1 &
```



## GRPO

```bash
CUDA_VISIBLE_DEVICES=0 python qwen_grpo.py
```



## ChatBot
```bash
python qwen_chat.py
```



## Evaluation

```bash
python qwen_eval.py --checkpoint-path <model_path> --eval_data_path <data_path>
```

