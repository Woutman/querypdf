FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cu126

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "./main/app.py", " --server.runOnSave true"]