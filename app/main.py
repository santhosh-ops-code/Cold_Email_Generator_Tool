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

    # ---------------- INPUTS ----------------
    url_input = st.text_input(
        "Enter a Job URL:",
        placeholder="https://careers.nike.com/job/xyz"
    )

    tone = st.selectbox(
        "Select email tone",
        ["Professional", "Friendly", "Formal"],
        index=0
    )

    col1, col2 = st.columns(2)
    generate_btn = col1.button("Generate Cold Email")
    regenerate_btn = col2.button("🔄 Regenerate")

    # ---------------- LOGIC ----------------
    if (generate_btn or regenerate_btn) and url_input:
        try:
            with st.spinner("Scraping job description and generating email..."):
                loader = WebBaseLoader([url_input])
                documents = loader.load()

                if not documents:
                    st.warning("Could not load the job page content.")
                    return

                # Limit content to avoid token explosion
                raw_text = " ".join(doc.page_content for doc in documents[:5])
                cleaned_data = clean_text(raw_text)

                jobs = llm.extract_jobs(cleaned_data)

                if not jobs:
                    st.warning(
                        "⚠️ No job information could be extracted.\n\n"
                        "**Reason:** Many job portals (Amazon, Google, LinkedIn) "
                        "are JavaScript-heavy and block scraping.\n\n"
                        "**Try instead:**\n"
                        "- Nike Careers\n"
                        "- Accenture Careers\n"
                        "- IBM Jobs\n"
                        "- Deloitte Careers\n"
                        "- Infosys / Zoho / TCS job pages"
                    )
                    return

                for idx, job in enumerate(jobs, start=1):
                    st.subheader(f"✉️ Cold Email #{idx}")

                    skills = job.get("skills", [])
                    portfolio_links = portfolio.query_links(skills)

                    email = llm.write_mail(
                        job=job,
                        links=portfolio_links
                    )

                    st.code(email, language="markdown")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


# ---------------- APP ENTRY ----------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="Cold Email Generator",
        page_icon="📧",
        layout="wide"
    )

    chain = Chain()
    portfolio = Portfolio()

    create_streamlit_app(chain, portfolio, clean_text)
