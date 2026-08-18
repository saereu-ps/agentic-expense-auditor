# Corporate Expense & Policy Auditor

An enterprise-grade, multi-agent orchestration system designed to strictly audit and synthesize corporate travel and expense policies using Retrieval-Augmented Generation (RAG).

## Project Overview
The Corporate Expense & Policy Auditor acts as an automated financial assistant. It securely searches a localized knowledge base containing corporate policies and synthesizes a comprehensive, professionally formatted answer for the end-user while strictly adhering to internal currency exchange rules.

## Architecture & Design Decisions
This project employs a Sequential Multi-Agent Workflow orchestrated via LangGraph. 

- **Agent 1: Data Retriever (The Auditor)**
  - Performs a semantic TF-IDF search against `knowledge_base.txt`.
  - **Constraint Compliance:** Built completely from scratch using standard Python libraries (`math`, `collections`, `re`) without external vector databases.
  - **Strict Information Extraction:** Uses a strict LLM filtering step to extract verbatim text snippets. It is explicitly programmed to never answer questions directly, preventing hallucinations and preserving data integrity.

- **Agent 2: Report Generator (The Synthesizer)**
  - Receives the raw chunks and leverages an LLM to synthesize the final response.
  - **Security & Safety:** Features strict prompt injection prevention and politely declines out-of-scope/unrelated queries.
  - **Strict Mathematics:** Enforces strict mathematical calculations of Thai Baht (THB) conversions based on embedded static historical exchange rates.
  - **Formatting Constraints:** Rigidly adheres to a Markdown structure with bolded currency/numbers and bulleted lists.

## Features & Highlights
- **Azure OpenAI Native integration:** Seamlessly supports Azure OpenAI deployments (e.g., `gpt-5-mini`) via environment variables, with a graceful fallback to Groq.
- **Enhanced UI (Streamlit):** Displays the user's exact query before synthesizing the policy results, utilizing clean Markdown for a professional look.

## Prerequisites
- Python 3.9+
- A valid LLM API Key (Azure OpenAI or Groq)

## Setup & Execution

1. **Configure Secrets**
   Copy the provided `.env.example` file to create your local `.env` configuration:
   ```bash
   cp .env.example .env
   ```
   Add your API keys to the `.env` file (Fill in `AZURE_OPENAI_API_KEY` or `GROQ_API_KEY`).

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
