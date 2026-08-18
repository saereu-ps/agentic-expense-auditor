import os
import logging
import streamlit as st
from agents import run_auditor

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Corporate Expense Auditor",
    layout="centered"
)

def main():
    st.markdown("""
    <style>
        h1, h2, h3, h4, p, li, span {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            letter-spacing: -0.2px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Corporate Expense & Policy Auditor")
    st.markdown("Ask a question about corporate travel policies, allowances, or expenses.")
    st.write("")
    
    with st.form("query_form"):
        query = st.text_input("Enter your question:", placeholder="e.g. What is the hotel allowance for Japan?")
        submitted = st.form_submit_button("Audit Policy")
    
    st.write("")
        
    if submitted and query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("Environment variables are not configured. Please contact the administrator.")
            st.stop()
            
        with st.spinner("Reviewing policies..."):
            try:
                result_state = run_auditor(query)
                retrieved_chunks = result_state.get("retrieved_chunks", [])
                final_response = result_state.get("final_response", "No response generated.")
                
                with st.expander("View Retrieved Policy Snippets"):
                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(f"**Snippet {i+1}:**\n> {chunk}")
                    else:
                        st.write("No relevant policy snippets found.")
                
                st.write("")
                st.divider()
                st.write("")
                
                with st.container(border=True):
                    st.markdown(final_response)
                
            except Exception as e:
                logger.error(f"UI caught an orchestration exception: {e}")
                st.error("We are currently experiencing technical difficulties. Please try again later.")

if __name__ == "__main__":
    main()
