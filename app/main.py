import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


# ---------- Helper ----------
def extract_company_name(text: str) -> str:
    lines = text.split("\n")
    for line in lines[:5]:
        if len(line.strip()) > 3:
            return line.strip()[:50]
    return "the company"


# ---------- Streamlit App ----------
def create_streamlit_app(llm, portfolio):
    st.title("📧 Cold Email Generator")
    st.caption(
        "Generate personalized cold emails by mapping job requirements "
        "to representative demo and academic projects."
    )

    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.company.com/job/xyz"
    )

    tone = st.selectbox(
        "Select email tone",
        ["Professional", "Friendly", "Concise", "Confident"]
    )

    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button("Generate Cold Email")
    with col2:
        regen_btn = st.button("🔁 Regenerate")

    if (generate_btn or regen_btn) and url_input:
        try:
            with st.spinner("Scraping job posting..."):
                loader = WebBaseLoader([url_input])
                documents = loader.load()

                raw_text = " ".join(doc.page_content for doc in documents[:5])
                cleaned_text = clean_text(raw_text)
                company_name = extract_company_name(cleaned_text)

            with st.spinner("Extracting job details..."):
                jobs = llm.extract_jobs(cleaned_text)

            if not jobs:
                st.warning("No job information could be extracted.")
                return

            portfolio.load_portfolio()

            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"✉️ Cold Email #{idx}")

                skills = job.get("skills", [])
                links = portfolio.query_links(skills)[:3]

                email = llm.write_mail(
                    job_description=job.get("description", str(job)),
                    links=links,
                    tone=tone,
                    company_name=company_name
                )

                # -------- Display structured email --------
                if "Subject:" in email:
                    subject, rest = email.split("Subject:", 1)[1].split("\n", 1)
                    st.markdown("### 📌 Subject")
                    st.write(subject.strip())
                    st.markdown("### ✉️ Email Body")
                    st.write(rest.strip())
                else:
                    st.code(email, language="markdown")

                st.download_button(
                    "⬇️ Download Email",
                    email,
                    file_name="cold_email.txt"
                )

        except Exception as e:
            st.error("An error occurred while generating the email.")
            st.exception(e)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cold Email Generator",
        page_icon="📧",
        layout="wide"
    )

    chain = Chain()
    portfolio = Portfolio()

    create_streamlit_app(chain, portfolio)
