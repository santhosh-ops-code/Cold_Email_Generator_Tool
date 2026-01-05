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
    You are Santhosh, an early-career professional from Santhosh AI Labs applying for the role below.

    STRICT RULES:
    - Do NOT invent years of experience.
    - Do NOT exaggerate seniority.
    - Use a realistic student / fresher / junior profile.
    - You MUST include a section titled "Portfolio Links".
    - Under "Portfolio Links", list links which are matching,relevant or closer to the job profile. if all are matching give 2-3 utmost
    - DOn;t give much spaces between the content make the e-mail look more realisitc
    - Do NOT summarize the links.
    - Do NOT omit the links. If no skills are matching then you can omit and summarize in 1-2 lines relevant to the jo profile.

    JOB DESCRIPTION:
    {job}

    PORTFOLIO LINKS (PRINT EXACTLY):
    {links}

    EMAIL TONE:
    {tone}

    Write a concise, professional cold email.

    End EXACTLY with:
    Best regards,
    Santhosh
    Santhosh AI Labs
    """
        )

        # 🔐 Flatten links safely (FIXES your error)
        flat_links = []
        for l in links:
            if isinstance(l, list):
                flat_links.extend([str(x) for x in l])
            else:
                flat_links.append(str(l))

        chain = prompt | self.llm
        res = chain.invoke({
            "job": str(job),
            "links": "\n".join(flat_links),
            "tone": tone
        })

        return res.content

