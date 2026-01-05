import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils.text_utils import clean_text
from utils.pdf_export import generate_pdf


def normalize_skills(skills):
    """Flatten + stringify skills safely"""
    normalized = []

    for s in skills:
        if isinstance(s, list):
            normalized.extend([str(x) for x in s])
        else:
            normalized.append(str(s))

    return list(set(normalized))


def create_streamlit_app(llm, portfolio):
    st.title("📧 Cold Email Generator")
    st.caption("Generate personalized cold emails from job postings")

    url = st.text_input("Enter a Job URL")

    tone = st.selectbox(
        "Select email tone",
        ["Professional", "Friendly", "Confident"]
    )

    col1, col2 = st.columns(2)
    generate = col1.button("Generate Cold Email")
    regenerate = col2.button("🔁 Regenerate")

    if (generate or regenerate) and url:
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()

            raw_text = " ".join(d.page_content for d in docs[:5])
            cleaned = clean_text(raw_text)

            jobs = llm.extract_jobs(cleaned)

            for idx, job in enumerate(jobs, 1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = normalize_skills(job.get("skills", []))
                links = portfolio.query_links(skills)

                email = llm.write_mail(
                    job=job,
                    links=links,
                    tone=tone
                )

                st.code(email, language="markdown")

                pdf = generate_pdf(email)
                st.download_button(
                    "⬇️ Export as PDF",
                    pdf,
                    file_name="cold_email.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")


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
