import requests
import json

hub_ip = '10.70.0.64'
api_url = f'http://{hub_ip}/hub/api'
token = 'e976480cd4dd4addab434134aa5d8e56' # Replace with your actual API token

headers = {
    'Authorization': f'token {token}',
    'Content-Type': 'application/json',
}

# Example 1: List all users
try:
    response = requests.get(f'{api_url}/users', headers=headers)
    response.raise_for_status() # Raise an exception for HTTP errors
    users = response.json()
    print("Users on JupyterHub:")
    for user in users:
        print(f"- {user['name']} (Admin: {user['admin']}, Server running: {user['server'] is not None})")
except requests.exceptions.RequestException as e:
    print(f"Error listing users: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response content: {e.response.text}")


# Example 2: Start a user's server (replace 'your_username' with an actual username)
user_to_start = 'akumar' # Replace with a valid username on your Hub
try:
    response = requests.post(f'{api_url}/users/{user_to_start}/server', headers=headers)
    response.raise_for_status()
    print(f"\nAttempted to start server for user: {user_to_start}")
    # You might want to poll the /users/{user}/server endpoint to check for 'ready' status
except requests.exceptions.RequestException as e:
    print(f"\nError starting server for {user_to_start}: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response content: {e.response.text}")

# # Example 3: Stop a user's server
# try:
#     response = requests.delete(f'{api_url}/users/{user_to_start}/server', headers=headers)
#     response.raise_for_status()
#     print(f"\nAttempted to stop server for user: {user_to_start}")
# except requests.exceptions.RequestException as e:
#     print(f"\nError stopping server for {user_to_start}: {e}")
#     if hasattr(e, 'response') and e.response is not None:
#         print(f"Response content: {e.response.text}")