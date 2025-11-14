import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

"""
Демо работы с шаблоном.
Шаблон для вопросам к боту-агроному.
Поддерживаемые переменные
city - название города
season - время года
user_question - собственно вопрос
"""

def demo(city: str, season: str, user_question: str) -> None:
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL", "gpt-5"),
        temperature=0,
        request_timeout=15
    )

    # Шаблон
    prompt = PromptTemplate.from_template(
        """
        Ты профессиональный агроном, мужчина 40 лет.
        Учитывай город проживания пользователя {city} и сезон {season}.
        Давай краткие ответы (2-3 предложения) по вопросу пользователя {user_question}.
        """
    )

    # Последовательность для запроса в ИИ
    chain = prompt | llm

    response = chain.invoke(
        {
            'city': city,
            'season': season,
            'user_question': user_question
        }
    )

    print(response.content)

if __name__ == "__main__":
    demo('Токио', 'Зима', 'Как вырастить томаты на подоконнике?')
