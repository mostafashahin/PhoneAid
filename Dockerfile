# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies for git and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gnupg \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Clone the GitHub repository (replace with your repo URL)
ARG REPO_URL=https://github.com/mostafashahin/PhoneAid.git
ARG BRANCH=main
RUN git clone --branch $BRANCH $REPO_URL /app
RUN git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI service using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

