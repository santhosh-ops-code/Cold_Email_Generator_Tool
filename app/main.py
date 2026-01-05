import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio):
    st.title("📧 Cold Email Generator")
    st.caption("Generate personalized cold emails directly from job postings")

    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.company.com/job/xyz"
    )

    generate_btn = st.button("Generate Cold Email")

    if generate_btn and url_input:
        try:
            with st.spinner("Scraping job page..."):
                loader = WebBaseLoader([url_input])
                documents = loader.load()

                raw_text = " ".join(
                    doc.page_content for doc in documents[:5]
                )

                cleaned_text = clean_text(raw_text)

            with st.spinner("Extracting job details..."):
                jobs = llm.extract_jobs(cleaned_text)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            portfolio.load_portfolio()

            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = job.get("skills", [])
                links = portfolio.query_links(skills)

                email = llm.write_mail(
                    job_description=job.get("description", str(job)),
                    links=links
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

    create_streamlit_app(chain, portfolio)
