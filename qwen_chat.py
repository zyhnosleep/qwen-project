import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

while True:
    user_input = input("question：")
    formatted_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128)
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    answer = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print("response：", answer)
