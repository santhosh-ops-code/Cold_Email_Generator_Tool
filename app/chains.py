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
            temperature=0.3,  # allows variation, avoids repetition
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        """
        Extract job information from scraped text.
        Always returns a list of job dictionaries.
        """

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The text above is from a job or careers webpage.

            Extract the job information and return JSON with these keys:
            - role
            - experience
            - skills (list)
            - description

            Rules:
            - Use ENGLISH only
            - If information is missing, infer reasonably
            - Return ONLY valid JSON
            - No explanations or extra text

            ### VALID JSON:
            """
        )

        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            parsed = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException(
                "Failed to parse job information. Page may block scraping."
            )

        # Always return list
        return parsed if isinstance(parsed, list) else [parsed]

    def write_mail(self, job, links):
        """
        Generate a cold email based on extracted job and portfolio links.
        """

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            {job_description}

            ### CONTEXT:
            You are Santhosh, an aspiring software and data professional.

            The portfolio links provided are representative demo and academic projects.
            They are meant to showcase relevant technical capabilities,
            not to claim official client work.

            ### TASK:
            Write a professional cold email to the hiring team:

            - Use ENGLISH only
            - Personalize the email based on the job role and skills
            - Reference the portfolio links naturally
            - Be concise, confident, and respectful
            - Avoid company names like AtliQ
            - Do NOT mention AI, LLMs, or automation
            - No preamble or explanation

            ### PORTFOLIO LINKS:
            {link_list}

            ### EMAIL:
            """
        )

        chain_email = prompt_email | self.llm
        response = chain_email.invoke(
            {
                "job_description": job,
                "link_list": links
            }
        )

        return response.content
