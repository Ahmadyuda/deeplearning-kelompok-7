# Gunakan image Python yang memiliki dependensi sistem yang kuat
FROM python:3.9-slim 

# Atur direktori kerja
WORKDIR /app

# Salin requirements.txt dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi
COPY . .

# Exposure port (standar Streamlit adalah 8501)
EXPOSE 8501

# Perintah untuk menjalankan Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true"]
