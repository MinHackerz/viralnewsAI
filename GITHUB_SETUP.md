# Setting up GitHub Actions for ViralNewsAI

For the automated workflow to run correctly, you need to set up the following secrets in your GitHub repository:

## Required Secrets

1. **FACEBOOK_ACCESS_TOKEN** - Your Facebook API access token
2. **NEWSAPI_KEY** - Your NewsData.io API key
3. **GOOGLE_API_KEY** - Your Google API key for Gemini AI

## How to Add Secrets to your GitHub Repository

1. Go to your GitHub repository
2. Click on "Settings" (tab at the top)
3. In the left sidebar, click on "Secrets and variables" then "Actions"
4. Click on "New repository secret"
5. Add each of the required secrets one by one:
   - Name: `FACEBOOK_ACCESS_TOKEN`
   - Value: (paste your Facebook access token)
   - Click "Add secret"
   
   Repeat for `NEWSAPI_KEY` and `GOOGLE_API_KEY`

## Verifying Secrets

You can verify that your secrets are correctly configured by manually running the "Check Environment Secrets" workflow:

1. Go to the "Actions" tab in your repository
2. Select "Check Environment Secrets" from the workflows list
3. Click "Run workflow"
4. Check the logs to ensure all secrets are properly detected

## Troubleshooting

If you receive an error about missing environment variables:

1. Make sure all required secrets are added exactly as named above
2. Check that the GitHub Actions workflow has permission to access these secrets
3. Verify that the secrets are not expired (especially for the Facebook token)

For more information on managing GitHub secrets, see the [GitHub documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets). 