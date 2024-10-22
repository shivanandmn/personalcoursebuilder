import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from sklearn.cluster import KMeans
from src.embed_courses import DBEmbedder
from src.db.db_transact import DBTransaction


class CourseRecommenderPipeline(DBEmbedder):
    def __init__(self, course_file_path=None, model_name="paraphrase-MiniLM-L12-v2"):
        super().__init__(course_file_path=None, model_name=model_name)
        self.db_transaction = DBTransaction(db_name="course_builder")

        # Quality filter map
        self.quality_map = {
            "very_high_quality": [
                "edX",
                "Swayam",
                "Udacity",
                "Coursera",
                "Pluralsight",
                "LinkedIn Learning",
                "FutureLearn",
                "MIT OpenCourseWare",
                "Cisco Networking Academy",
                "AWS Skill Builder",
                "Microsoft Learn",
                "The Odin Project",
                "Kaggle",
                "Exercism",
                "Google Cloud Skills Boost",
                "Qwiklabs",
                "Skillshop",
                "Stanford OpenEdx",
            ],
            "high_quality": [
                "freeCodeCamp",
                "Udemy",
                "Trailhead",
                "Codecademy",
                "Treehouse",
                "Scrimba",
                "Zero To Mastery",
                "DataCamp",
                "Test Automation University",
                "TryHackMe",
                "PentesterAcademy",
                "Brilliant",
                "Jovian",
            ],
            "medium_quality": [
                "Skillshare",
                "MasterClass",
                "Kadenze",
                "openSAP",
                "HubSpot Academy",
                "Domestika",
                "CreativeLive",
                "SymfonyCasts",
                "The Great Courses Plus",
                "Wolfram U",
                "Semrush Academy",
                "Craftsy",
                "Designlab",
                "MongoDB University",
                "Complexity Explorer",
            ],
            "lower_quality": [
                "OpenLearn",
                "Cybrary",
                "YouTube",
                "Independent",
                "Saylor Academy",
                "egghead.io",
                "Frontend Masters",
                "Study.com",
                "California Community Colleges System",
                "Cognitive Class",
                "Edureka",
                "Laracasts",
                "openHPI",
                "Open2Study",
                "iversity",
                "Polimi OPEN KNOWLEDGE",
                "OpenLearning",
                "Canvas Network",
            ],
        }
        # Convert quality map to numerical scores
        self.quality_scores = {
            provider: score
            for score, providers in enumerate(
                [
                    self.quality_map["lower_quality"],
                    self.quality_map["medium_quality"],
                    self.quality_map["high_quality"],
                    self.quality_map["very_high_quality"],
                ],
                start=1,
            )
            for provider in providers
        }

    def calculate_similarity(self, skills):
        # Create embeddings for the provided skills
        embedding_skills = self.model.encode(skills, show_progress_bar=False)

        # Calculate cosine similarity between courses and skills
        similarities = []
        total_courses = {}
        for vector in embedding_skills:
            similar = self.db_transaction.get_based_on_knn(
                vector.tolist(), k=50, collection="course_data"
            )
            similarities.append(similar)
            for d in similar:
                if d["_id"] not in total_courses:
                    total_courses[d["_id"]] = d
        # self.similarity_df = pd.DataFrame(
        #     cosine_similarities, index=self.embedding_ids, columns=skills
        # )

        return pd.DataFrame(list(total_courses.values()))

    def filter_courses(self, data, user_preferences):
        # Apply user filters
        df_filtered = data[
            (data["duration_hrs"] >= user_preferences["min_duration"])
            & (data["duration_hrs"] <= user_preferences["max_duration"])
            & (data["avg_rating"] >= user_preferences["min_rating"])
            & (data["language"].str.lower() == user_preferences["language"].lower())
        ]
        return df_filtered

    def apply_quality_scores(self, df_filtered):
        df_filtered["quality_score"] = (
            df_filtered["provider"].map(self.quality_scores).fillna(0)
        )
        max_score = max(self.quality_scores.values())
        df_filtered["quality_score"] = df_filtered["quality_score"] / max_score
        return df_filtered

    def get_weighted_score(
        self,
        df_filtered,
        skill_weight=0.5,
        rating_weight=0.3,
        duration_weight=0.2,
        quality_weight=0.2,
    ):
        # Calculate weighted score
        df_filtered = self.apply_quality_scores(df_filtered)
        df_filtered["weighted_score"] = (
            skill_weight
            * df_filtered["score"]
            # + rating_weight * df_filtered["avg_rating"]
            # + duration_weight * (1 / (df_filtered["duration_hrs"] + 1))
            # + quality_weight * df_filtered["quality_score"]
        )
        return df_filtered.sort_values(by="weighted_score", ascending=False)

    def deduplicate_courses(self, df_sorted, num_clusters=10):
        """
        Deduplicate courses using KMeans clustering on embeddings.
        Select the highest weighted score course from each cluster.
        """
        # Apply KMeans clustering using filtered embeddings
        print("Applying KMeans clustering for deduplication...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(np.array([x for x in df_sorted["vector"].values]))
        df_sorted["cluster"] = cluster_labels

        # Select the highest weighted score course from each cluster
        df_deduplicated = df_sorted.loc[
            df_sorted.groupby("cluster")["weighted_score"].idxmax()
        ]

        return df_deduplicated.reset_index(drop=True)

    def recommend_courses(self, skills, user_preferences):
        similarity_df = self.calculate_similarity(skills)
        similarity_df["duration_hrs"] = similarity_df["duration"].apply(self.parse_time_string)
        df_filtered = self.filter_courses(similarity_df, user_preferences)

        # Filter courses with skill similarity
        df_filtered = df_filtered[df_filtered["score"] > 0.5].dropna(
            how="all", axis=0
        )
        # filtered_similarity_df = (
        #     filtered_similarity_df.sum(axis=1)
        #     .reset_index()
        #     .rename(columns={0: "similarity_score", "index": "_id"})
        # )
        # df_filtered = df_filtered[
        #     df_filtered["_id"].isin(filtered_similarity_df["_id"])
        # ]
        # df_filtered = df_filtered.merge(filtered_similarity_df, on="_id")

        # Get weighted score and recommend top courses
        df_sorted = self.get_weighted_score(df_filtered)
        # save_all_recommendations_path = self.save_path + "_rec_courses.csv"
        # df_sorted.reset_index(drop=True).to_csv(save_all_recommendations_path)
        # Extract filtered embeddings for deduplication
        # filtered_embeddings = [
        #     self.embedding_courses[np.where(self.embedding_ids == _id)[0][0]]
        #     for _id in df_filtered["_id"]
        # ]

        # Deduplicate courses using KMeans clustering
        df_deduplicated = self.deduplicate_courses(
            df_sorted, num_clusters=10
        )
        # save_deduplicated_recommendations_path = (
        #     self.save_path + "_dedup_rec_courses.csv"
        # )
        # df_deduplicated.to_csv(save_deduplicated_recommendations_path, index=False)

        return {
            "all_recommendations": df_sorted,
            "dedup_recommendations": df_deduplicated,
        }  # .head(user_preferences["top_n"])


# Example usage
if __name__ == "__main__":
    # User preferences
    user_preferences = {
        "min_duration": 1,
        "max_duration": 40,
        "min_rating": 4.0,
        "language": "English",
        "top_n": 20,
    }

    # Skills list
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

    # Initialize the pipeline and get recommendations
    course_recommender = CourseRecommenderPipeline(
        "./data/raw/course/courses_103190.csv"
    )
    recommended_courses = course_recommender.recommend_courses(skills, user_preferences)

    print(recommended_courses)
