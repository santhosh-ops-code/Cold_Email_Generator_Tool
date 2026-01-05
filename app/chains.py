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
            temperature=0.3,  # avoids repetition but stays professional
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            Extract job information from the above content.

            Return a JSON object (or list of objects) with EXACTLY these keys:
            - role
            - experience
            - skills
            - description

            If some fields are missing, make a reasonable assumption.
            Return ONLY valid JSON. No explanations.

            ### JSON:
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            jobs = parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Failed to parse job information.")

        return jobs if isinstance(jobs, list) else [jobs]

    def write_mail(self, job_description: str, links: list[str]) -> str:
        prompt_email = PromptTemplate.from_template(
            """
            You are Santhosh, an AI & Data Science Consultant at Santhosh AI Labs.

            Santhosh AI Labs builds intelligent, data-driven solutions using
            AI, automation, and analytics.

            NOTE:
            The portfolio links provided are representative demo and academic
            projects showcasing capability, not client work.

            ### JOB DESCRIPTION:
            {job_description}

            ### RELEVANT PORTFOLIO LINKS:
            {link_list}

            ### INSTRUCTION:
            Write a professional, personalized cold email for this role.
            Guidelines:
            - Do NOT mention any company unless stated in the job description
            - Do NOT invent clients or experience
            - Keep it concise and role-specific
            - Highlight relevant skills and value
            - End with a polite call-to-action
            - Avoid generic or repetitive phrasing

            ### EMAIL:
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description,
            "link_list": links
        })

        return res.content


if __name__ == "__main__":
    print("GROQ API KEY LOADED:", bool(os.getenv("GROQ_API_KEY")))
