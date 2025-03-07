import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from typing import List, Dict
import boto3
from botocore.exceptions import ClientError

class PDFGenerator:
    def __init__(self, filename: str):
        self.filename = filename
        self.doc = SimpleDocTemplate(self.filename, pagesize=letter)
        self.styles = getSampleStyleSheet()

        # Wasabi S3 Configuration
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ.get("WASABI_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("WASABI_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("WASABI_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("WASABI_REGION")
        )
        self.bucket_name = os.environ.get("WASABI_BUCKET_NAME")
        if not all([self.bucket_name, self.s3_client]):
           raise ValueError("Wasabi S3 environment variables not set.")

    def generate_report(self, title: str, query: str, result: str, source_info: List[Dict]):
        """Generates a PDF report with ReportLab and uploads to Wasabi S3."""
        elements = []

        # Title
        elements.append(Paragraph(title, self.styles['h1']))
        elements.append(Spacer(1, 0.2*inch))

        # Query
        elements.append(Paragraph(f"<b>Query:</b> {query}", self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        # Result
        elements.append(Paragraph(f"<b>Result:</b>", self.styles['Normal']))
        elements.append(Paragraph(result, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))

        # Source Information
        elements.append(Paragraph("<b>Source Information:</b>", self.styles['h2']))
        for source in source_info:
            elements.append(Paragraph(f"Source ID: {source.get('source_id', 'N/A')}", self.styles['Normal']))
            elements.append(Paragraph(f"Source Type: {source.get('source_type', 'N/A')}", self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        try:
            # Build PDF
            self.doc.build(elements)
            # Upload to Wasabi
            self.upload_to_s3(self.filename)
            print(f"PDF generated and uploaded to Wasabi S3: {self.filename}")

        except ClientError as e:
            print(f"S3 Upload Error: {e}")
        except Exception as e:
            print(f"Error generating or uploading PDF: {e}")

        finally:
            if os.path.exists(self.filename):
                os.remove(self.filename)

    def upload_to_s3(self, filename: str):
        """Uploads a file to the configured Wasabi S3 bucket."""
        try:
            self.s3_client.upload_file(filename, self.bucket_name, filename)
            print(f"File '{filename}' uploaded to Wasabi S3 bucket '{self.bucket_name}'")
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise
