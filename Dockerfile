# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY .streamlit .
#COPY .env .
COPY app.py .

# ADD chroma_db_uploaded/ .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# --server.port 8501: Streamlit's default port
# --server.enableCORS false: Often needed in containerized environments
# --server.enableXsrfProtection false: Often needed in containerized environments
# --server.headless true: Prevents Streamlit from trying to open a browser on the server
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.headless", "true"]

