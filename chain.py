import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from resume import MY_RESUME

CANDIDATE_NAME = "Denis" 

CLASSIFIER_TEMPLATE = """You are a strict text classifier. Your ONLY job is to determine 
if the following text is a real job vacancy/job posting.

A real job vacancy typically contains MOST of these:
- A job title (e.g. "Senior Developer", "Data Analyst", "Researcher")
- Company or team description
- List of responsibilities or duties
- List of requirements or qualifications
- Mention of specific technologies, tools, or skills needed

The following are NOT job vacancies:
- Horoscopes, astrology (even if they mention "career" or "salary")
- News articles, blog posts
- Motivational text, self-help advice
- Recipes, stories, poems
- Random text, greetings, questions

TEXT TO CLASSIFY:
{vacancy}

Respond with ONLY one word: JOB or NOT"""


TEMPLATE = """You are a witty high-energy talent agent representing  a developer named {name}.
 Your goal is to SELL this candidate to the recruiter. You are NOT neutral. You are biased and confident.

RESUME:
{resume}

JOB VACANCY:
{vacancy}


RULES:
- Language: same as the job vacancy
- Tone: confident, punchy, no corporate bullshit
- Name: refer to candidate ONLY as "{name}", never use surname


OUTPUT FORMAT:
     LINE 1: "🔥 MATCH SCORE: [X]/10" (Be generous but realistic.).
     LINE 2: Empty line.
     LINE 3: "VERDICT:" followed by 2-3 short, punchy paragraphs explaining why {name} destroys this role.
             - Focus ONLY on matches. 
             - IGNORE missing skills (do not mention what he lacks).
             - Use bullet points for key matches.
     
   - **Constraints**:
     - Keep it SHORT (max 150 words).
     - NO "advice for cover letter".
     - NO "areas for improvement".
     - NO "Best regards".
"""

REJECTION_TEMPLATE = """The user pasted some random text instead of a job vacancy.
Write a short (1-2 sentences), witty, slightly sarcastic response.
Make fun of the situation. Tell them to paste a real job vacancy.
Write in the same language as the text below.

TEXT:
{vacancy}
"""
def _get_llm():
    return ChatGroq(
        temperature=0.3,
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY")
    )

def _classify(vacancy_text: str) -> bool:
    prompt = PromptTemplate(
        input_variables=["vacancy"],
        template=CLASSIFIER_TEMPLATE
    )
    chain = prompt | _get_llm()
    response = chain.invoke(input={"vacancy": vacancy_text})
    result = response.content.strip().upper()
    return "JOB" in result and "NOT" not in result


def _get_rejection(vacancy_text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["vacancy"],
        template=REJECTION_TEMPLATE
    )
    chain = prompt | _get_llm()
    response = chain.invoke(input={"vacancy": vacancy_text})
    return response.content


def _get_match(vacancy_text: str) -> str:
    prompt = PromptTemplate(
        input_variables=["resume", "vacancy", "name"],
        template=MATCH_TEMPLATE
    )
    chain = prompt | _get_llm()
    response = chain.invoke(input={
        "resume": MY_RESUME,
        "vacancy": vacancy_text,
        "name": CANDIDATE_NAME
    })
    return response.content


def analyze_vacancy(vacancy_text: str) -> str:
    is_job = _classify(vacancy_text)
    if not is_job:
        return _get_rejection(vacancy_text)

    return _get_match(vacancy_text)