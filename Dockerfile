FROM public.ecr.aws/lambda/python:3.10

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy function code
COPY app3.py .

# Set Lambda handler
CMD ["app3.lambda_handler"]