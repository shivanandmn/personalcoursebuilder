from dotenv import load_dotenv
import google.generativeai as genai
import os
import json
from langchain_core.output_parsers import JsonOutputParser
from copy import deepcopy

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


class LLMGen:
    def __init__(self, model_name="gemini-1.5-flash", generation_config=None) -> None:
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=model_name, generation_config=generation_config
        )

    def generate_json_content(self, content: str | list, generation_config=None):
        response = self.model.generate_content(
            content, generation_config=generation_config
        )
        try:
            parser = JsonOutputParser()
            result = parser.parse(response.text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            result = None
        return result, response.to_dict()

    def generate_content(self, content, generation_config):
        for i in range(3):
            try:
                gen_config = deepcopy(generation_config)
                response = self.model.generate_content(
                    content, generation_config=gen_config
                )
                gen_config["temperature"] += 0.09
                print("response :", response)
                response_text = response.text
                response = response.to_dict()
                response["retries"] = i
                return response_text, response
            except:
                pass
        return {}, {}
