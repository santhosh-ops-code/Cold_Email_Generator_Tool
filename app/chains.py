import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        prompt = PromptTemplate.from_template(
            """
            You are given scraped job text.

            Extract job info as JSON with keys:
            role, experience, skills (list), description.

            TEXT:
            {page_data}

            Return ONLY valid JSON.
            """
        )

        chain = prompt | self.llm
        result = chain.invoke({"page_data": cleaned_text})

        try:
            parsed = JsonOutputParser().parse(result.content)
        except OutputParserException:
            raise ValueError("Failed to parse job data")

        return parsed if isinstance(parsed, list) else [parsed]

    def write_mail(self, job, links, tone):
        prompt = PromptTemplate.from_template(
            """
            You are Santhosh, a candidate applying for the role below.

            JOB:
            {job}

            PORTFOLIO LINKS:
            {links}

            TONE: {tone}

            Write a professional cold email.
            Do NOT mention demo or fake projects.
            Keep it realistic and concise.
            """
        )

        chain = prompt | self.llm
        res = chain.invoke({
            "job": job,
            "links": "\n".join(links),
            "tone": tone
        })

        return res.content
