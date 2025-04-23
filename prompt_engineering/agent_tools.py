import requests
from langchain.tools import Tool

def fetch_all_users():
    url = "https://your-api.com/users"  # 🔁 Replace with your API
    response = requests.get(url)
    response.raise_for_status()
    return response.text

get_users_tool = Tool(
    name="GetAllUsers",
    func=lambda _: fetch_all_users(),
    description="Use this tool to fetch all users from the internal API."
)

# You can add more tools here as needed.
tools = [get_users_tool]
