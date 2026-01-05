import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Email Generator")
    st.caption("Generate personalized cold emails directly from job postings")

    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.company.com/job/xyz"
    )

    submit_button = st.button("Generate Cold Email")

    if submit_button and url_input:
        try:
            loader = WebBaseLoader([url_input])
            documents = loader.load()

            raw_text = " ".join(doc.page_content for doc in documents[:5])
            data = clean_text(raw_text)

            jobs = llm.extract_jobs(data)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = job.get("skills", [])
                links = portfolio.query_links(skills)

                email = llm.write_mail(
                    job_description=job,
                    portfolio_links=links,
                    job_url=url_input
                )

                st.code(email, language="markdown")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cold Email Generator",
        page_icon="📧",
        layout="wide"
    )

    chain = Chain()
    portfolio = Portfolio()

    create_streamlit_app(chain, portfolio, clean_text)

