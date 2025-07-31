import os

import requests


def get_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    return [f"{model['id']}" for model in response.json()["data"]]

