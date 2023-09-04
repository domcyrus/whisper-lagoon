FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

# Run updates and install dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential && \
    apt-get install -y ffmpeg && \
    apt-get install -y sox libsox-dev python3-dev python3-pip python3-distutils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy and install the requirements
COPY ./requirements.txt /requirements.txt

# Pip install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

ENV MODEL=large-v2
ENV QUANTIZATION=float16

# Copy the current directory contents into the container at /app
COPY main.py /app/main.py
COPY download.py /app/download.py

# Set the working directory to /app
WORKDIR /app

# Expose a port for the server
EXPOSE 8000

# Run the app
CMD uvicorn main:app --host 0.0.0.0