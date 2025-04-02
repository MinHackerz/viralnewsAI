import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print the loaded environment variables
print("GOOGLE_API_KEY:", os.getenv('GOOGLE_API_KEY'))
print("FACEBOOK_ACCESS_TOKEN:", os.getenv('FACEBOOK_ACCESS_TOKEN') is not None)
print("NEWSAPI_KEY:", os.getenv('NEWSAPI_KEY')) 