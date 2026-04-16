from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/distill/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B", split="train").select(range(10))

model_id = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/DeepSeek-R1-Distill-Qwen-7B"  # 这里最好用r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:
    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )
if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)