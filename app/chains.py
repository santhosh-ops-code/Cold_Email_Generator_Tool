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
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The scraped text is from a job/careers page.
            Extract job information and return JSON with these keys:
            role, experience, skills, description.

            Return ONLY valid JSON.
            ### JSON:
            """
        )

        chain = prompt_extract | self.llm
        response = chain.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            result = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Failed to parse job data.")

        return result if isinstance(result, list) else [result]

    def write_mail(self, job, links):
        tone = job.get("tone", "Professional")

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            {job_description}

            ### INSTRUCTION:
            You are Santhosh, a software & data professional.
            Write a {tone} cold email applying for the above role.

            Important rules:
            - This is NOT client work
            - Portfolio links are demo/academic projects
            - Be honest, realistic, and concise
            - Do NOT mention AtliQ or any company name
            - Do NOT exaggerate experience

            Include the most relevant portfolio links:
            {link_list}

            ### EMAIL:
            """
        )

        chain = prompt_email | self.llm

        response = chain.invoke({
            "job_description": str(job),
            "link_list": "\n".join(links),
            "tone": tone
        })

        return response.content.strip()
