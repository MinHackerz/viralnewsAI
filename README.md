# Viral News Automation

Welcome to the Viral News Automation project! This guide will help you set up and run the project seamlessly.

## Setup Guide

1. **Clone the Repository or Create a New PyCharm Project**:
   - Clone the repository using:
     ```sh
     git clone https://github.com/MinHackerz/viralnewsAI.git
     ```
   - Or, create a new project in PyCharm and link it to the repository.

2. **Python Compatibility**:
   - This project is compatible with Python 3.11 and 3.12
   - Python 3.13 is supported with the latest updates
   - Python 3.10 or earlier may have compatibility issues

3. **Install Dependencies**:
   - Navigate to the project directory and install the required dependencies:
     ```sh
     pip install -r requirements.txt
     ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the root directory of the project.
   - Add the following API keys to the `.env` file:
     ```env
     FACEBOOK_ACCESS_TOKEN=your_facebook_access_token
     GOOGLE_API_KEY=your_google_api_key
     NEWSAPI_KEY=your_newsapi_key
     CLOUDFLARE_API_KEY=your_cloudflare_api_key
     ```

5. **Generate Facebook Access Token**:
   - To generate a `FACEBOOK_ACCESS_TOKEN`, follow these steps:

     ### Step 1: Create a Facebook App
     - Go to the [Meta for Developers](https://developers.facebook.com/) website and log in with your Facebook account.
     - Create a new app or use an existing one.
     - Note down your `App ID` and `App Secret` from the app dashboard.

     ### Step 2: Get a Short-Lived Access Token
     - Redirect the user to Facebook's OAuth dialog to grant permissions:
       ```bash
       https://www.facebook.com/v18.0/dialog/oauth?
         client_id={app-id}
         &redirect_uri={redirect-uri}
         &response_type=code
         &scope=pages_show_list,pages_manage_posts,pages_read_engagement
       ```
       Replace `{app-id}` with your App ID and `{redirect-uri}` with your redirect URI.

     - After authorization, Facebook will redirect to your `redirect_uri` with a `code` parameter. Use this code to get a short-lived access token:
       ```bash
       curl -X GET "https://graph.facebook.com/v18.0/oauth/access_token?
         client_id={app-id}
         &redirect_uri={redirect-uri}
         &client_secret={app-secret}
         &code={code}"
       ```

     ### Step 3: Exchange for a Long-Lived Access Token
     - Extend the short-lived token to a long-lived token:
       ```bash
       curl -X GET "https://graph.facebook.com/v18.0/oauth/access_token?
         grant_type=fb_exchange_token
         &client_id={app-id}
         &client_secret={app-secret}
         &fb_exchange_token={short-lived-token}"
       ```

     ### Step 4: Get Page Access Token
     - List pages and get the page access token:
       ```bash
       curl -X GET "https://graph.facebook.com/v18.0/me/accounts?access_token={long-lived-token}"
       ```
     - Copy the `access_token` for your page and add it to your `.env` file.

6. **Add Facebook Page ID**:
   - Open the `viralnewsAI.py` file.
   - Locate the section where the Facebook Page ID is required.
   - Add your Facebook Page ID to the script.

7. **Run the Script**:
   - Execute the main script to start the automation:
     ```sh
     python viralnewsAI.py
     ```

## Running in PyCharm

- **Open the Project**:
  - Open the project folder in **PyCharm**.

- **Install Dependencies**:
  - Use the integrated **Terminal** in PyCharm to install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

- **Configure Environment Variables**:
  - Ensure your `.env` file is correctly set up with the required environment variables.

- **Run the Script**:
  - Run the script using PyCharm's run configuration or directly from the terminal within PyCharm.

## GitHub Actions Automation

This project includes GitHub Actions workflows to automate posting:

1. **Setting Up GitHub Actions**:
   - See [GITHUB_SETUP.md](GITHUB_SETUP.md) for detailed instructions on configuring GitHub Actions
   - You must add your API keys as GitHub repository secrets

2. **Automation Schedule**:
   - By default, the workflow runs every hour
   - You can modify the schedule in `.github/workflows/main.yaml`

3. **Troubleshooting**:
   - If you encounter environment variable errors, verify your GitHub secrets are correctly set up
   - Check the GitHub Actions logs for detailed error information

## Sample Output

Here is a sample output generated by the Viral News Automation script:

![Sample Output](images/sample_output.jpg)
