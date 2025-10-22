conda create -n llmapp python=3.11 -y

conda activate llmapp

pip install -r requirements.txt

streamlit run streamlit_app.py

