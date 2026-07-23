# Dockerfile
FROM python:3.9-slim

# Set up a working directory inside the container
WORKDIR /app

# Copy ONLY requirements first (to leverage Docker's layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project files 
# (anything not excluded by .dockerignore)
COPY . .

# The user should only run main.py. So let's set it as the default entry point
CMD ["python", "main.py"]
