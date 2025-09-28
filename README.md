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

seo-keyword-ai-agent/
│── src/
│ ├── app.py # Master pipeline orchestrator
│ ├── ranking.py # Keyword discovery & scoring
│ ├── postprocess.py # Cleans & enriches results
│ ├── dashboard.py # Streamlit visualization
│ ├── server.py # FastAPI/Render server
│── output/ # Generated keyword results
│── .env # API keys (not committed)
│── requirements.txt # Python dependencies
│── README.md # Project documentation




