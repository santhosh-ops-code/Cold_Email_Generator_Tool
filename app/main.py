from utils.pdf_export import generate_pdf
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio):
    st.title("📧 Cold Email Generator")
    st.caption("Generate personalized cold emails from job postings")

    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.company.com/job/xyz"
    )

    tone = st.selectbox(
        "Select email tone",
        ["Professional", "Friendly", "Confident"]
    )

    col1, col2 = st.columns([1, 1])
    generate_btn = col1.button("Generate Cold Email")
    regenerate_btn = col2.button("🔁 Regenerate")

    if (generate_btn or regenerate_btn) and url_input:
        try:
            loader = WebBaseLoader([url_input])
            documents = loader.load()

            raw_text = " ".join(doc.page_content for doc in documents[:5])
            cleaned_text = clean_text(raw_text)

            jobs = llm.extract_jobs(cleaned_text)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = job.get("skills", [])
                links = portfolio.query_links(skills)

                email = llm.write_mail(
                    job=job,
                    links=links,
                    tone=tone
                )

                st.code(email, language="markdown")
                pdf = generate_pdf(email)

                st.download_button(
                    label="⬇️ Export as PDF",
                    data=pdf,
                    file_name="cold_email.pdf",
                    mime="application/pdf"
                )


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
    portfolio.load_portfolio()

    create_streamlit_app(chain, portfolio)
