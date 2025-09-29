import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    # Get the API key from environment variables
    api_key = os.getenv("SERPAPI_KEY")
    
    if api_key:
        print("✅ Project setup complete!")
        print(f"API key loaded: {api_key[:5]}...")
    else:
        print("❌ Warning: API_KEY not found in environment variables")
        print("Make sure you have a .env file with your API_KEY")

if __name__ == "__main__":
    main()