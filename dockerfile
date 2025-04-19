# use python 3.9 slim as base image
FROM python:3.9-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (using --without-dev instead of --no-dev)
RUN poetry install --without dev --no-interaction --no-ansi

# copy project files
COPY App/app.py /app/

# create model directory
RUN mkdir -p /app/Trained_models

# copy model files
COPY Trained_models/pii_detector_model_bert /app/Trained_models/pii_detector_model_bert

# expose port
EXPOSE 8000

# set environment variable
ENV PYTHONUNBUFFERED=1

# start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
