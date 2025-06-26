FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN curl -L "https://huggingface.co/KevinLuis/modelAPI/resolve/main/Identifica_Sala_Ocupada.zip" -o Identifica_Sala_Ocupada.zip
RUN unzip Identifica_Sala_Ocupada.zip

EXPOSE 5000
CMD ["python", "app.py"]
