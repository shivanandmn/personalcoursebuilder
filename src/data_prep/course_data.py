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
print()
