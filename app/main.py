import streamlit as st
from dotenv import load_dotenv
import os

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# -------------------------------------------------
# Basic Streamlit setup
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Cold Email Generator",
    page_icon="📧"
)

st.title("📧 Cold Mail Generator")
st.write("Generate personalized cold emails from job postings.")

load_dotenv()  # local use; Streamlit Cloud uses Secrets


# -------------------------------------------------
# UI Inputs
# -------------------------------------------------
url_input = st.text_input(
    "Enter a Job URL:",
    value="https://jobs.nike.com/job/R-33460"
)

submit_button = st.button("Generate Cold Email")


# -------------------------------------------------
# Button Action (IMPORTANT: heavy logic here only)
# -------------------------------------------------
if submit_button:
    if not url_input:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing job posting and generating email..."):
            try:
                # 🔥 IMPORTS INSIDE BUTTON (CRITICAL)
                from langchain_community.document_loaders import WebBaseLoader

                # Initialize core components
                chain = Chain()
                portfolio = Portfolio()

                # Load website content
                loader = WebBaseLoader(url_input)
                page_content = loader.load().pop().page_content
                data = clean_text(page_content)

                # Load portfolio data
                portfolio.load_portfolio()

                # Extract jobs
                jobs = chain.extract_jobs(data)

                if not jobs:
                    st.warning("No job roles found on the page.")
                else:
                    for idx, job in enumerate(jobs, start=1):
                        skills = job.get("skills", [])
                        links = portfolio.query_links(skills)
                        email = chain.write_mail(job, links)

                        st.subheader(f"✉️ Cold Email #{idx}")
                        st.code(email, language="markdown")

            except Exception as e:
                st.error("❌ An error occurred while generating the email.")
                st.exception(e)
