import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.4,  # ✅ avoids repeated outputs
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
            Extract the job posting and return JSON with:
            - role
            - experience
            - skills
            - description

            Return ONLY valid JSON. No explanations.

            ### JSON:
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Failed to parse job data.")

        return parsed if isinstance(parsed, list) else [parsed]

    def write_mail(self, job_description: str, links: list[str]) -> str:
        prompt_email = PromptTemplate.from_template(
            """
            You are a professional job applicant.

            ### JOB DESCRIPTION:
            {job_description}

            ### PORTFOLIO LINKS:
            {link_list}

            ### INSTRUCTION:
            Write a personalized cold email for this job.
            Rules:
            - Do NOT mention any company unless present in the job description
            - Do NOT use fake company names
            - Professional, concise tone
            - Highlight matching skills
            - End with a polite call-to-action

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
