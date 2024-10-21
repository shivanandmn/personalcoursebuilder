from google.cloud import storage
from PIL import Image
import io
import os

import toml
import json

# Load the TOML file
with open('config.toml', 'r') as toml_file:
    toml_data = toml.load(toml_file)

json_data = json.dumps({key: value for key, value in toml_data.items() if key not in ["GOOGLE_API_KEY", "MONGODB_PASSWORD"]}, indent=4)

# Save the JSON data to a file
with open('config.json', 'w') as json_file:
    json_file.write(json_data)

# Initialize the Google Cloud Storage Client
client = storage.Client.from_service_account_json(
    "config.json"
)


class GCPStorage:

    def __init__(self, bucket_name: str) -> None:
        self.bucket = client.bucket(bucket_name)

    def upload_file(self, file_path, destination_blob_name=None, folder_path=""):
        """Uploads a file to the GCP bucket."""
        if destination_blob_name is None:
            destination_blob_name = file_path.split("/")[-1]
        destination_blob_name = os.path.join(folder_path, destination_blob_name)
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        return blob.public_url  # Returns the public URL of the uploaded file

    def upload_image(self, image: Image, destination_blob_name, format="JPEG"):
        """Uploads an Image object directly to GCP bucket."""
        blob = self.bucket.blob(destination_blob_name)
        byte_stream = io.BytesIO()
        image.save(
            byte_stream, format=format
        )  # Save image to a BytesIO object in the specified format
        byte_stream.seek(0)  # Move to the beginning of the BytesIO object
        blob.upload_from_file(byte_stream, content_type=f"image/{format.lower()}")
        return blob.public_url  # Returns the public URL of the uploaded image

    def download_file(self, blob_name, destination_file_path):
        """Downloads a file from GCP bucket."""
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(destination_file_path)
        return destination_file_path  # Returns the path to the downloaded file

    def delete_file(self, blob_name):
        """Deletes a file from GCP bucket."""
        blob = self.bucket.blob(blob_name)
        blob.delete()
        return blob_name  # Returns the name of the deleted file

    def list_files(self, prefix=None):
        """Lists all files in GCP bucket optionally filtered by a prefix."""
        blobs = self.bucket.list_blobs(
            prefix=prefix
        )  # List all files in the bucket that start with the prefix
        return [blob.name for blob in blobs]  # Returns a list of file names


if __name__ == "__main__":
    gcp_storage = GCPStorage(bucket_name="course_builder_dataset")
    gcp_storage.upload_file(
        "./data/raw/job_desc/AI-ML Engineer.pdf", "job_description/AI-ML Engineer.pdf"
    )
    print(gcp_storage.list_files())
