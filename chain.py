import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from resume import MY_RESUME

TEMPLATE = """You are a career matching expert.

RESUME:
{resume}

JOB VACANCY:
{vacancy}

Your task:
1. Explain WHY this candidate is a great fit
2. Rate the match from 1 to 10
3. Suggest what to highlight in a cover letter

Be enthusiastic but honest. Write in the same language as the vacancy."""


def analyze_vacancy(vacancy_text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["resume", "vacancy"],
        template=TEMPLATE
    )
    llm = ChatGroq(
        temperature=0.3,
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    chain = prompt | llm
    response = chain.invoke(input={
        "resume": MY_RESUME,
        "vacancy": vacancy_text
    })
    return response.content