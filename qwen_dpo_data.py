import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from distilabel.llms import OpenAILLM
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration, EvolQuality, UltraFeedback
from tqdm import tqdm
from utils.utils import find_files

TMP_PATH        = "/archive/share/cql/aaa/tmp"
DATA_PATH       = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data"
DATA_FILES      = find_files(["rl"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
MODEL_A_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen3-8B-Instruct"
MODEL_B_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Meta-Llama-3-8B-Instruct-Chinese"
TRAIN_SUBSET    = 20
DS_KEY         = "xxx"

ds = load_dataset("parquet", data_files=DATA_FILES,
                        split="train", cache_dir=TMP_PATH)
if TRAIN_SUBSET > 0:
    ds = ds.select(range(TRAIN_SUBSET))
questions = ds["question"]

llm_a = TransformersLLM(model=MODEL_A_PATH)
llm_a.load()
llm_b = TransformersLLM(model=MODEL_B_PATH)
llm_b.load()

judge_llm = OpenAILLM(
    model="deepseek-chat",
    base_url=r"https://api.deepseek.com/v1",
    api_key=DS_KEY
)
judge_llm.load()

evol_quality   = EvolQuality(llm=judge_llm, num_evolutions=1) 
evol_quality.load()

ultrafeedback  = UltraFeedback(llm=judge_llm)           
ultrafeedback.load()

records = []

def chat(msg): 
    return [{"role": "user", "content": msg}]

GEN_KWARGS = dict(max_new_tokens=256, temperature=0.7)

for q in tqdm(questions, desc="Generating/Rewrite/Scoring"):
    # (i) 两个模型独立生成
    gen_a = llm_a.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
    gen_b = llm_b.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]

    # (ii) EvolQuality 迭代 1 次
    evo_a = evol_quality.process([{"instruction": q, "response": gen_a}])
    evo_b = evol_quality.process([{"instruction": q, "response": gen_b}])
    evo_a = next(evo_a)[0]["evolved_response"]
    evo_b = next(evo_b)[0]["evolved_response"]

    # (iii) UltraFeedback 打分
    feedback = ultrafeedback.process([{
        "instruction": q,
        "generations": [evo_a, evo_b]
    }])
    feedback = next(feedback)
    ratings     = feedback[0]["ratings"]
    rationales  = feedback[0]["rationales"]

    records.append({
        "question":q,
        "original_response_j":gen_a,
        "original_response_k":gen_b,
        "response_j":evo_a,
        "response_k":evo_b,
        "rating_j":ratings[0],
        "rating_k":ratings[1],
        "rationales":rationales
    })

df = pd.DataFrame(records)
df.to_parquet(os.path.join(DATA_PATH, "dpo_data.parquet"), index=False)