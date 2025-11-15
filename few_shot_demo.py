import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import yaml

load_dotenv()

"""
Демо работы few-shot
"""

def demo(function_name: str) -> None:
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL", "gpt-5"),
        temperature=0,
        request_timeout=15
    )

    # YAML
    with open("few_shot_prompts.yaml", "r", encoding="utf-8") as f:
        yaml_prompts = yaml.safe_load(f)

    # Шаблон для примеров
    example_prompt = PromptTemplate.from_template(yaml_prompts['prompts']['example'])

    # Few-shot шаблон
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=yaml_prompts['prompts']['examples'],
        example_prompt=example_prompt,
        prefix=yaml_prompts['prompts']['example_prefix'],
        suffix=yaml_prompts['prompts']['example_suffix'],
        input_variables=['input']
    )

    # Подготавливаем и выводим
    formatted_prompt = few_shot_prompt_template.format(input=function_name)
    response = llm.invoke(formatted_prompt)
    print(response.content)

if __name__ == "__main__":
    demo('tuple')
