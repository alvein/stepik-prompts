import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

"""
Демо работы с шаблоном.
Шаблон для вопросам к боту-агроному.
Поддерживаемые переменные
city - название города
season - время года
user_question - собственно вопрос
"""

class Advise(BaseModel):
    question_summary: str = Field(description='Краткий пересказ вопроса от пользователя')
    plant: str = Field(description='Название растения')
    season: str = Field(description='Время года')
    city: str = Field(description='Местоположение')
    advice: str = Field(description='Совет')


def demo(city: str, season: str, user_question: str) -> None:
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL", "gpt-5"),
        temperature=0,
        request_timeout=15
    )

    # Парсер данных
    output_parser = PydanticOutputParser(pydantic_object=Advise)
    format_instructions = output_parser.get_format_instructions()

    # Шаблон
    prompt = PromptTemplate.from_template(
        """
        Ты профессиональный агроном, мужчина 40 лет.
        Учитывай город проживания пользователя {city} и сезон {season}.
        {format_instructions}
        Давай краткие ответы (2-3 предложения) по вопросу: {user_question}.
        """,
        partial_variables={
            'format_instructions': format_instructions
        }
    )

    chain = prompt | llm | output_parser

    try:
        response = chain.invoke(
            {
                'city': city,
                'season': season,
                'user_question': user_question
            }
        )
        print(response.model_dump_json())

    except:
        print({"error": "Invalid response format"})

if __name__ == "__main__":
    demo('Токио', 'Зима', 'Как вырастить томаты на подоконнике?')
