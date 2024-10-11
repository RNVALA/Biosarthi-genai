# Use the official AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory inside the container
WORKDIR /var/task

# Copy the application code for the Lambda function
COPY app.py .  

# Install the dependencies directly
RUN pip install --no-cache-dir langchain langchain-core langchain-community faiss-cpu pymupdf aws-lambda-powertools langchain-aws pypdf

# Print environment variables and file structure for debugging
RUN echo "=== Environment Variables ===" && env
RUN echo "=== Filesystem Structure ===" && ls -R /var/task

# Command to run the Lambda handler when the container starts
CMD ["app.lambda_handler"]

