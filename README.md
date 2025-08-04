# Corrective-RAG-with-Langraph

To run locally

git clone https://github.com/Regu7/Corrective-RAG-with-Langraph.git

Provide your API keys in OPENAI and Tavily API keys in .env file. Use Gituhub secrets if your pushing to github.

conda create -n adaptive_rag python=3.12 -y

conda activate adaptive_rag

streamlit run app.py

Upload your files in the left side box and click process, now you can start asking questions to the chatbot

---------------------------

Docker Image Creation

docker build -t adaptive_rag:latest .

docker run --name  adaptive_rag_cont -p 8501:8501 adaptive_rag:latest