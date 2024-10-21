import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from bson import ObjectId
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()


def connect_to_mongodb():
    uri = f"mongodb+srv://shivanandnaduvin:{os.getenv('MONGODB_PASSWORD')}@cluster0.rt3b5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # Replace with your MongoDB URI

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi("1"))

    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client


class CourseDataSave:
    def __init__(
        self,
        db="course_builder",
        collection="course_data",
    ) -> None:
        self.client = connect_to_mongodb()
        self.collection = self.client[db][collection]

    def get_by_course_id(self, course_id):
        data = list(self.collection.find({"_id": course_id}))
        if len(data) == 0:
            return None
        return data[0]

    def insert_one(self, data):
        if self.get_by_course_id(data["_id"]) is None:
            now = datetime.now()
            data["timestamp"] = str(int(now.timestamp()))
            data["datetime"] = str(now)
            return self.collection.insert_one(data)
        else:
            print(f"Course with ID {data['_id']} already exists in the database.")

    def insert_many_from_pd(self, df):
        try:

            # Convert the DataFrame to a list of dictionaries for MongoDB
            data_list = df.to_dict(orient="records")

            # Insert each row in the collection
            for data in tqdm(data_list, total=len(data_list), desc="Inserting"):
                self.insert_one(data)

            print(f"Inserted {len(data_list)} records from CSV to MongoDB.")
        except Exception as e:
            print(f"Error inserting records from CSV: {e}")

    def delete_many(self, document_ids):
        object_ids = [ObjectId(doc_id) for doc_id in document_ids]
        query = {"_id": {"$in": object_ids}}
        result = self.collection.delete_many(query)
        print(f"Documents deleted: {result.deleted_count}")

    def update_one(self, data, filter_key="id"):
        """
        Must element in data dict: id
        """
        if self.get_by_course_id(data[filter_key]) is not None:
            # Document filter (criteria for the document you want to update)
            document_filter = {filter_key: data[filter_key]}
            del data[filter_key]
            # New key-value pair you want to add
            new_data = {"last_modified": str(int(datetime.now().timestamp()))}
            new_data.update(data)

            # Update the document
            self.collection.update_one(document_filter, {"$set": new_data})
        else:
            print("Exception: Document not found")


if __name__ == "__main__":
    df = pd.read_csv("./data/raw/course/courses_103190.csv")

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

    df = df.rename(columns=map_cols)
    df = df[list(map_cols.values())]

    course_data_save = CourseDataSave()
    course_data_save.insert_many_from_pd(df)
