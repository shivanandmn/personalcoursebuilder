from src.db.mongodb import connect_to_mongodb
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from tqdm import tqdm


class DBEmbedder:
    def __init__(self, course_file_path, model_name="paraphrase-MiniLM-L12-v2"):
        self.client = connect_to_mongodb()
        self.db = self.client["course_builder"]
        self.collection = self.db["course_data"]
        self.embedding_courses = None
        self.embedding_ids = None
        if course_file_path is not None:
            self.save_path = f"./data/output/{os.path.basename(course_file_path).split('.')[0]}"
            # Load and preprocess course data
            self.df = self._load_and_preprocess_data(course_file_path)

        # Initialize the model for embedding generation
        self.model = SentenceTransformer(model_name)


    def _load_and_preprocess_data(self, file_path):
        # Load dataset
        df = pd.read_csv(file_path)

        # Rename columns for consistency
        map_cols = {
            "course_id": "_id",
            "course_name": "name",
            "description": "description",
            "duration": "duration",
            "cost": "pricing",
            "course_level": "level",
            "course_provider": "provider",
            "course_num_rating": "num_rating",
            "course_avg_rating": "avg_rating",
            "rating": "rating",
            "language": "language",
            "course_certificate": "certificate",
            "course_subject": "subject",
            "course_type": "type",
        }

        # Rename and filter columns
        df = df.rename(columns=map_cols)
        df = df[list(map_cols.values())]
        df = df[df["language"] == "English"]
        df["duration_hrs"] = df["duration"].apply(self.parse_time_string)

        return df

    @staticmethod
    def parse_time_string(time_str):
        if "On-Demand" in time_str:
            return (
                0  # or some default value like a specific number of hours if you prefer
            )

        weeks = days = hours = minutes = 0

        # Regular expressions for matching different time components
        week_match = re.search(r"(\d+)-?(\d+)?\s*weeks?", time_str)
        day_match = re.search(r"(\d+)-?(\d+)?\s*days?", time_str)
        hour_match = re.search(r"(\d+)-?(\d+)?\s*hours?", time_str)
        minute_match = re.search(r"(\d+)-?(\d+)?\s*minutes?", time_str)

        # Extract weeks
        if week_match:
            weeks = (
                int(week_match.group(1))
                if not week_match.group(2)
                else (int(week_match.group(1)) + int(week_match.group(2))) // 2
            )

        # Extract days
        if day_match:
            days = (
                int(day_match.group(1))
                if not day_match.group(2)
                else (int(day_match.group(1)) + int(day_match.group(2))) // 2
            )

        # Extract hours
        if hour_match:
            hours = (
                int(hour_match.group(1))
                if not hour_match.group(2)
                else (int(hour_match.group(1)) + int(hour_match.group(2))) // 2
            )

        # Extract minutes
        if minute_match:
            minutes = (
                int(minute_match.group(1))
                if not minute_match.group(2)
                else (int(minute_match.group(1)) + int(minute_match.group(2))) // 2
            )

        # Convert everything to total hours
        total_hours = weeks * 7 * 24 + days * 24 + hours + minutes / 60
        return total_hours

    def create_embeddings(self, columns_to_encode):
        # Create embeddings for courses using specified columns
        embedding_save_path = self.save_path + "_embeddings.npy"
        embedding_ids_save_path = self.save_path + "_embedding_ids.npy"
        if os.path.exists(embedding_save_path) and os.path.exists(
            embedding_ids_save_path
        ):
            print("Loading precomputed course embeddings...")
            self.embedding_courses = np.load(embedding_save_path, allow_pickle=True)
            self.embedding_ids = np.load(embedding_ids_save_path, allow_pickle=True)
        else:
            print("Computing course embeddings...")
            courses = (
                self.df[columns_to_encode]
                .apply(lambda x: "\t".join(x.astype(str)), axis=1)
                .tolist()
            )
            self.embedding_courses = self.model.encode(courses, show_progress_bar=True)
            self.embedding_ids = self.df["_id"].values
            os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)
            np.save(embedding_save_path, self.embedding_courses)
            np.save(embedding_ids_save_path, self.embedding_ids)

    def embed_and_store(self, columns_to_encode=["name", "description"]):
        # Step 1: Generate embeddings
        print("Generating embeddings...")
        self.create_embeddings(columns_to_encode)
        print("Generated!")
        if self.embedding_courses is None:
            raise ValueError(
                "Embeddings have not been created yet. Please call create_embeddings first."
            )
        self.embedding_ids = self.df["_id"].values

        # Step 2: Insert embeddings into MongoDB
        print("Inserting embeddings into MongoDB...")
        for _id, embedding in tqdm(zip(self.embedding_ids, self.embedding_courses), total=len(self.embedding_ids)):
            document = {
                "_id": int(_id),
                "embedding_keys": columns_to_encode,
                "vector": embedding.tolist(),  # Convert the embedding to a list so it can be stored in MongoDB
            }

            # Use update_one to either update an existing document or insert if it doesn't exist
            self.collection.update_one(
                {"_id": int(_id)},
                {"$set": document},
                upsert=True  # Create the document if it doesn't exist
            )

        print("Embeddings successfully inserted into MongoDB.")

# Example usage
if __name__ == "__main__":

    embedder = DBEmbedder("./data/raw/course/courses_103190.csv")
    embedder.embed_and_store()
