import requests


def generate_incident_report(vehicle_id):

    prompt = f"""
    Generate a SHORT real-time traffic violation alert.

    Rules:
    - Maximum 2 sentences
    - Do NOT invent license plate, location, or fines
    - Mention vehicle ID
    - Mention line crossing violation
    - Professional concise format

    Vehicle ID: {vehicle_id}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"]
