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
    git clone [https://github.com/omraghu07/seo-keyword-ai-agent.git](https://github.comomraghu07/seo-keyword-ai-agent.git)
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

 2. Launch the Interactive Dashboard
Visualize the results using the Streamlit dashboard.

Bash

streamlit run src/dashboard.py
3. Run as an API (Render/FastAPI)
Deploy the application as a web service using Gunicorn.

Bash

gunicorn -k uvicorn.workers.UvicornWorker src.server:app --bind 0.0.0.0:8000 --workers 2
🔗 n8n Integration
Automate your keyword analysis by creating an n8n workflow.

Start a new workflow with a Webhook node.

Connect the webhook to the deployed Render API endpoint.

Configure the HTTP Request node to send a POST request to https://seo-keyword-ai-agent.onrender.com/analyze with a JSON body:

JSON

{
  "seed": "global internship",
  "top": 10
}
Add Email or Slack nodes to automatically send reports after the workflow completes.

📊 Example Output
Here is a sample of the top 5 keyword opportunities identified by the agent:

Keyword	Volume	Competition	Score	Results
UCLA Global Internship Program	2000	0.03	30.12	0
Summer Internship Programs - CIEE	1666	0.33	9.26	54,000
Global Internship Program HENNGE	2000	0.35	9.01	110,200
Berkeley Global Internships Paid	1666	0.45	6.98	219,000
Global Internship Remote	2500	0.50	6.66	174,000,000

Export to Sheets
📌 Roadmap
[x] Core pipeline (keyword discovery + analysis)

[x] Dashboard for visualization

[x] Deployment on Render

[x] n8n integration for automation

[ ] Add real Google Ads search volume API integration

[ ] Implement multi-language keyword support

[ ] Add AI-powered keyword clustering

👨‍💻 Author
Om Raghuvanshi

An engineering student passionate about AI, Generative AI, and Travel Filmmaking.

Portfolio: [Your Website/LinkedIn]

Twitter: [@yourhandle]

Contact: [your email]

⚡ If you find this project useful, don’t forget to ⭐ star the repository and fork it!
