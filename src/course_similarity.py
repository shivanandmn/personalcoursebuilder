import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os


class CourseSkillMatcher:
    def __init__(self, course_file_path, model_name="paraphrase-MiniLM-L12-v2"):
        # Load dataset
        self.df = self._load_and_preprocess_data(course_file_path)

        # Initialize the Sentence Transformer model
        self.model = SentenceTransformer(model_name)

        # Set embedding save path based on the input file name
        self.embedding_save_path = f"./data/output/{os.path.basename(course_file_path).split('.')[0]}_embeddings.npy"

        # Embedding placeholders
        self.embedding_skills = None
        self.embedding_courses = None
        self.similarity_df = None

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
            if week_match.group(2):
                weeks = (int(week_match.group(1)) + int(week_match.group(2))) // 2
            else:
                weeks = int(week_match.group(1))

        # Extract days
        if day_match:
            if day_match.group(2):
                days = (int(day_match.group(1)) + int(day_match.group(2))) // 2
            else:
                days = int(day_match.group(1))

        # Extract hours
        if hour_match:
            if hour_match.group(2):
                hours = (int(hour_match.group(1)) + int(hour_match.group(2))) // 2
            else:
                hours = int(hour_match.group(1))

        # Extract minutes
        if minute_match:
            if minute_match.group(2):
                minutes = (int(minute_match.group(1)) + int(minute_match.group(2))) // 2
            else:
                minutes = int(minute_match.group(1))

        # Convert everything to total hours
        total_hours = weeks * 7 * 24 + days * 24 + hours + minutes / 60
        return total_hours

    def create_embeddings(self, columns_to_encode):
        # Create embeddings for courses using specified columns
        print("Creating embeddings for courses using specified columns...")

        # Check if course embeddings already exist
        if os.path.exists(self.embedding_save_path):
            print("Loading precomputed course embeddings...")
            self.embedding_courses = np.load(self.embedding_save_path)
        else:
            print("Computing course embeddings...")
            courses = (
                self.df[columns_to_encode]
                .apply(lambda x: "\t".join(x.astype(str)), axis=1)
                .tolist()
            )
            self.embedding_courses = self.model.encode(courses, show_progress_bar=True)
            # Save embeddings to file
            os.makedirs(os.path.dirname(self.embedding_save_path), exist_ok=True)
            np.save(self.embedding_save_path, self.embedding_courses)

    def calculate_cosine_similarity(self, skills):
        # Create embeddings for the provided skills
        self.embedding_skills = self.model.encode(skills, show_progress_bar=False)

        # Calculate cosine similarity between courses and skills
        if self.embedding_skills is None or self.embedding_courses is None:
            raise ValueError(
                "Embeddings have not been created yet. Please call create_embeddings first."
            )

        print("Calculating cosine similarities...")
        cosine_similarities = cosine_similarity(
            self.embedding_courses, self.embedding_skills
        )

        # Create a DataFrame to hold the similarity values
        self.similarity_df = pd.DataFrame(
            cosine_similarities, index=self.df["_id"], columns=skills
        )

    def get_similarity_dataframe(self):
        if self.similarity_df is None:
            raise ValueError(
                "Cosine similarity has not been calculated yet. Please call calculate_cosine_similarity first."
            )
        return self.similarity_df

    def run(self, skills, columns_to_encode):
        # End-to-end method to create embeddings and calculate similarity
        self.create_embeddings(columns_to_encode)
        self.calculate_cosine_similarity(skills)
        return self.get_similarity_dataframe()


if __name__ == "__main__":
    # List of skills
    skills = [
        "Probability",
        "Statistics",
        "Linear Algebra",
        "Programming",
        "Machine Learning",
        "NLP tasks",
        "Topic modelling",
        "Entity Extraction",
        "Summarization",
        "Sentiment analysis",
        "Object detection",
        "Image segmentation",
        "Image classification",
        "AWS",
        "Azure",
        "Google Cloud",
        "Cloud AI services",
        "Cloud AI tools",
        "Python",
        "R",
        "TensorFlow",
        "PyTorch",
        "scikit-learn",
        "MLOps",
    ]

    # Columns to encode
    columns_to_encode = ["name", "description"]

    # Initialize CourseSkillMatcher
    course_skill_matcher = CourseSkillMatcher("./data/raw/course/courses_103190.csv")

    # Run the end-to-end process
    similarity_df = course_skill_matcher.run(skills, columns_to_encode)

    # Display similarity DataFrame
    print(similarity_df.head())
