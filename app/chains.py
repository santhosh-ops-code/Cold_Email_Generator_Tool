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
            ### SCRAPED JOB PAGE TEXT:
            {page_data}

            ### INSTRUCTION:
            Extract job information from the text.
            Return ONLY valid JSON in the following format:
            [
              {{
                "role": "",
                "experience": "",
                "skills": [],
                "description": ""
              }}
            ]

            ### RULES:
            - Respond ONLY in English
            - No explanations
            - No markdown
            - No extra text
            """
        )

        chain = prompt | self.llm
        response = chain.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            parsed = parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse job details")

        return parsed if isinstance(parsed, list) else [parsed]

    def write_mail(self, job, links, tone):
        # ✅ SAFELY FORMAT PORTFOLIO LINKS
        formatted_links = []
        for item in links:
            if isinstance(item, (list, tuple)):
                formatted_links.append(f"{item[1]}: {item[0]}")
            else:
                formatted_links.append(str(item))

        prompt = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            {job}

            ### PORTFOLIO PROJECTS (Demo / Academic):
            {links}

            ### TONE:
            {tone}

            ### INSTRUCTION:
            You are Santhosh, an aspiring professional reaching out regarding the above role.

            Write a realistic cold email:
            - Use ONLY English
            - Do NOT claim client work
            - Treat portfolio links as demo/academic projects
            - Professional, concise, and relevant
            - Include a subject line
            - No preamble text

            ### EMAIL:
            """
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "job": str(job),
                "links": "\n".join(formatted_links),
                "tone": tone
            }
        )

        return response.content.strip()
