FROM python:3.7.3

WORKDIR flask_mask_app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .