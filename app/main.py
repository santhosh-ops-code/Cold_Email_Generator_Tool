import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils.text_utils import clean_text
from utils.pdf_export import generate_pdf


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

    # Session state for caching
    if "jobs" not in st.session_state:
        st.session_state.jobs = None

    # First-time generation
    if generate_btn and url_input:
        try:
            loader = WebBaseLoader([url_input])
            docs = loader.load()

            raw_text = " ".join(d.page_content for d in docs[:5])
            cleaned_text = clean_text(raw_text)

            jobs = llm.extract_jobs(cleaned_text)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            st.session_state.jobs = jobs

        except Exception as e:
            st.error(f"Error while fetching job data: {e}")
            return

    # Regenerate without scraping again
    if regenerate_btn and st.session_state.jobs:
        pass

    # Render emails
    if st.session_state.jobs:
        for idx, job in enumerate(st.session_state.jobs, start=1):
            st.subheader(f"✉️ Cold Email #{idx}")

            skills = job.get("skills", [])
            links = portfolio.query_links(skills)

            job_with_tone = {
                **job,
                "tone": tone
            }

            email = llm.write_mail(
                job=job_with_tone,
                links=links
            )

            st.code(email, language="markdown")

            pdf = generate_pdf(email)
            st.download_button(
                label="⬇️ Export as PDF",
                data=pdf,
                file_name=f"cold_email_{idx}.pdf",
                mime="application/pdf"
            )


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
