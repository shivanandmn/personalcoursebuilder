from pydantic import BaseModel, Field
from enum import Enum


def extract_pydantic_elements(obj):
    if obj is None:
        return ""
    # Handle lists of Pydantic models
    if isinstance(obj, list):
        elements = [extract_pydantic_elements(item) for item in obj]
        return "; ".join(elements)
    # Convert the object to a dictionary
    if not  isinstance(obj, dict):
        obj_dict = obj.dict()
    else:
        obj_dict = obj
    # Extract key-value pairs and concatenate them into a string
    elements = []
    for key, value in obj_dict.items():
        # If the value is another Pydantic model, recursively extract it
        if isinstance(value, BaseModel):
            elements.append(extract_pydantic_elements(value))
        # If the value is an Enum, use its value instead of the name
        elif isinstance(value, Enum):
            elements.append(f"{key}: {value.value}")
        else:
            elements.append(f"{key}: {value}")
    return ", ".join(elements)
