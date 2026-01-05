import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Cold Email Generator")
    st.caption(
        "Generate personalized cold emails by mapping job requirements "
        "to representative demo and academic projects."
    )

    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.company.com/job/xyz"
    )

    submit_button = st.button("Generate Cold Email")

    if submit_button:
        if not url_input.strip():
            st.warning("Please enter a valid job URL.")
            return

        try:
            # 1️⃣ Load webpage
            loader = WebBaseLoader([url_input])
            documents = loader.load()

            if not documents:
                st.error("Unable to load content from the given URL.")
                return

            # 2️⃣ Combine page text safely
            raw_text = " ".join(
                doc.page_content for doc in documents[:5] if doc.page_content
            )

            cleaned_data = clean_text(raw_text)

            # 3️⃣ Extract jobs using LLM
            jobs = llm.extract_jobs(cleaned_data)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            # 4️⃣ Generate cold emails
            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = job.get("skills", [])
                portfolio_links = portfolio.query_links(skills)

                # ✅ IMPORTANT: positional arguments (matches chains.py)
                email = llm.write_mail(job, portfolio_links)

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
