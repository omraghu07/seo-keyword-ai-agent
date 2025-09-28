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



---

## ⚙️ Installation  

1. **Clone the repo**  
```bash
git clone https://github.com/your-username/seo-keyword-ai-agent.git
cd seo-keyword-ai-agent


Create virtual environment

python -m venv agent_venv
source agent_venv/bin/activate   # Mac/Linux
agent_venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Setup .env file

SERPAPI_KEY=your_serpapi_key_here

▶️ Usage
Run the full pipeline
python src/app.py "global internship" --max-candidates 100 --top-results 50

Launch the dashboard
streamlit run src/dashboard.py

Run as an API (Render/FastAPI)
gunicorn -k uvicorn.workers.UvicornWorker src.server:app --bind 0.0.0.0:8000 --workers 2

🔗 n8n Integration

Create an n8n workflow with a Webhook node.

Connect it to the Render API:

POST https://seo-keyword-ai-agent.onrender.com/analyze
{
  "seed": "global internship",
  "top": 10
}


Add Email/Slack nodes to auto-send reports.

📊 Example Output

Top 5 Keyword Opportunities:

Keyword	Volume	Competition	Score	Results
UCLA Global Internship Program	2000	0.0	330.12	0
Summer Internship Programs - CIEE	1666	0.33	9.26	54,000
Global Internship Program HENNGE	2000	0.35	9.01	10,200
Berkeley Global Internships Paid	1666	0.45	6.98	219,000
Global Internship Remote	2500	0.50	6.66	174M
🛠️ Tech Stack

Python (Core language)

SerpAPI (Google search results API)

Pandas, Requests, Tabulate (Data processing)

Streamlit + Plotly (Dashboard & charts)

FastAPI + Gunicorn (API server)

Render (Deployment)

n8n (Workflow automation)

📌 Roadmap

 Core pipeline (keyword discovery + analysis)

 Dashboard for visualization

 Deployment on Render

 n8n integration for automation

 Add real Google Ads search volume API

 Multi-language keyword support

 AI-powered keyword clustering

👨‍💻 Author

Om Raghuvanshi – Engineering student passionate about AI, Generative AI, and Travel Filmmaking.

🌐 Portfolio: [Your Website/LinkedIn]

🐦 Twitter: [@yourhandle]

📧 Contact: [your email]

⚡ If you like this project, don’t forget to ⭐ star the repo and fork it!




