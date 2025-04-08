# AI Video Editor Backend

FastAPI backend for the AI Video Editor application.

## Deployment to Railway

### Prerequisites

- A [Railway](https://railway.app/) account
- [Railway CLI](https://docs.railway.app/develop/cli) (optional)

### Deployment Steps

1. **Push your code to GitHub**

   Make sure your backend code is pushed to the GitHub repository:
   ```
   git push -u origin main
   ```

2. **Deploy to Railway**

   #### Option 1: Deploy via Railway Dashboard

   1. Log in to [Railway](https://railway.app/)
   2. Click "New Project" > "Deploy from GitHub repo"
   3. Select your repository
   4. Railway will automatically detect the Python project and deploy it

   #### Option 2: Deploy via Railway CLI

   1. Install Railway CLI: `npm i -g @railway/cli`
   2. Login: `railway login`
   3. Link to your project: `railway link`
   4. Deploy: `railway up`

3. **Set Environment Variables**

   In the Railway dashboard:
   1. Go to your project
   2. Click on "Variables"
   3. Add the following environment variables:
      - `API_HOST`: 0.0.0.0
      - `PORT`: (Railway sets this automatically)
      - `SUPABASE_URL`: Your Supabase URL
      - `SUPABASE_KEY`: Your Supabase key
      - Other variables from your .env file as needed

4. **Configure Storage**

   For production, you should use Supabase or another cloud storage solution instead of local file storage. Update the `storage.py` file accordingly.

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   python run.py --reload
   ```

## API Documentation

Once deployed, you can access the API documentation at:
- Swagger UI: `https://your-railway-url/docs`
- ReDoc: `https://your-railway-url/redoc`
