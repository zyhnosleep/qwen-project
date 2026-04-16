import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from utils.grpo_utils import format_reward,accuracy_reward
from utils.utils import find_files,print_trainable_parameters,format_to_r1

# 设置环境变量以优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"

output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/grpo-zero"
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
data_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reasoning"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print_trainable_parameters(model)

# 加载数据集并进行预处理
directories = ["data"]
data_files = find_files(directories,"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reasoning")
dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["problem", "solution"], cache_dir=TMP_PATH) 
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.08).values()

train_dataset = train_dataset.map(format_to_r1)
train_dataset.remove_columns(["problem"])
valid_dataset = valid_dataset.map(format_to_r1)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = GRPOConfig(
    output_dir=output_path,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    bf16=True,
    max_completion_length=256,  
    num_generations=8,  
    per_device_train_batch_size = 24,
    max_prompt_length=512,  
    report_to="wandb",
    logging_steps=10,
    save_strategy="steps",
    save_steps=10,
)
if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-deepseek-grpo-zero",name="q2.5-0.5B-r1-grpo-zero"
    )

# 初始化Trainer
trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset
)

# 开始训练
print("Training...")
trainer.train()
trainer.save_model()  
tokenizer.save_pretrained(output_path) 