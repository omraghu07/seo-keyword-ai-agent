# SEO Keyword Research AI Agent  

An **AI-powered SEO keyword research agent** that discovers, analyzes, and ranks keyword opportunities using **SerpAPI**, with an interactive **Streamlit dashboard** for visualization and an **n8n integration** for automation.  

This project was built to demonstrate skills in **Python, AI agents, API integration, data visualization, and deployment** (Render + n8n).  

---

## 🚀 Features  

- 🔍 **Keyword Discovery** – Finds related keywords for any seed keyword.  
- 📊 **Keyword Analysis** – Scores keywords based on search volume, competition, and SERP signals.  
- 📂 **Data Export** – Saves results to CSV/Excel with metadata.  
- 📈 **Interactive Dashboard** – Streamlit + Plotly for keyword trends, competition heatmaps, and intent analysis.  
- 🤖 **AI Agent Workflow** – Automates tasks like keyword research → processing → reporting.  
- 🔗 **n8n Integration** – Trigger workflows via webhooks (e.g., run keyword research and auto-send results to Slack/Email).  
- 🌐 **Deployment** – Hosted on **Render** for API and dashboard access.  

---

## 🏗️ Project Structure  

## seo-keyword-ai-agent/
│
├── app.py # Master pipeline orchestrator

│──  dashboard.py # Streamlit visualization

│── src/

│ ├── postprocess.py # Cleans & enriches results

│ ├── ranking.py # Keyword discovery & scoring

│ ├── server.py # FastAPI/Render server

│── output/ # Generated keyword results

│── .env # API keys (not committed)

│── requirements.txt # Python dependencies

│── README.md # Project documentation



---

## ⚙️ Installation  

---

## ⚙️ Installation

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/your-username/seo-keyword-ai-agent.git](https://github.com/your-username/seo-keyword-ai-agent.git)
    cd seo-keyword-ai-agent
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv agent_venv
    
    # Mac/Linux
    source agent_venv/bin/activate
    
    # Windows
    agent_venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup .env file**
    Create a `.env` file in the root directory and add your API key:
    ```
    SERPAPI_KEY=your_serpapi_key_here
    ```

---
▶️ Usage
Run the full pipeline

Bash

python src/app.py "global internship" --max-candidates 100 --top-results 50
Launch the dashboard

Bash

streamlit run src/dashboard.py
Run as an API (Render/FastAPI)

Bash

gunicorn -k uvicorn.workers.UvicornWorker src.server:app --bind 0.0.0.0:8000 --workers 2
