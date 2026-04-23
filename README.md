# HMR-OCR-AI-Extractor
This project is a serverless AI-powered solution that extracts Hour Meter Readings (HMR) from machine images using image preprocessing and multimodal AI.

The application is deployed as an API using AWS Lambda and processes images sent from a frontend or external systems.

The system is packaged as a Docker container image, pushed to Amazon ECR, and used to create a container-based Lambda function. Each API request triggers the function, which processes the image and returns structured output.

The system consists of two main stages:

Image Preprocessing  
The input image is validated, auto-rotated using EXIF data, optionally deskewed using OpenCV, and resized for cost and performance optimization. This ensures the image is clean and readable before being processed by the AI model.

AI-Based Data Extraction  
The processed image is sent to a GPT-based multimodal model, which directly interprets the visual content and extracts the Hour Meter Reading (HMR) as structured JSON output.

Unlike traditional OCR pipelines, this approach does not rely on separate OCR engines. Instead, the AI model understands both the visual and contextual information in the image, enabling more robust extraction even in challenging conditions.

The system includes strict validation to ensure only clearly visible readings are accepted. If the reading is unclear or invalid, the API returns a user-friendly error message prompting for a better image.
