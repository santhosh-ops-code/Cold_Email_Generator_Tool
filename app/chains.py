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
            temperature=0.4,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        prompt = PromptTemplate.from_template(
            """
            ### SCRAPED JOB PAGE TEXT:
            {page_data}

            ### INSTRUCTION:
            Extract job information and return JSON with keys:
            role, experience, skills (list), description.

            Return ONLY valid JSON.
            """
        )

        chain = prompt | self.llm
        response = chain.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            parsed = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Failed to parse job data")

        return parsed if isinstance(parsed, list) else [parsed]

    def write_mail(self, job, links, tone):
        prompt = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            {job_data}

            ### PORTFOLIO LINKS:
            {portfolio_links}

            ### TONE:
            {tone}

            ### INSTRUCTION:
            You are Santhosh, an aspiring professional.
            Write a cold email applying for the role above.

            Rules:
            - Language: English only
            - Tone must match selection
            - Mention portfolio links naturally
            - Do NOT claim real client work
            - No preamble

            ### EMAIL:
            """
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "job_data": str(job),
            "portfolio_links": "\n".join(links),
            "tone": tone
        })

        return result.content
