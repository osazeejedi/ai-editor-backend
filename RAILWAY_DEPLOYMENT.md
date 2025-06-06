# Deploying AI Video Editor Backend to Railway

Railway is an excellent choice for deploying your FastAPI backend because:
1. It supports Python applications natively
2. It provides automatic HTTPS
3. It has built-in CI/CD with GitHub integration
4. It offers easy environment variable management

## Prerequisites

- A [Railway](https://railway.app/) account
- A [Supabase](https://supabase.com/) account (for production storage)
- Your code pushed to GitHub

## Step 1: Prepare Your Repository

We've created two different deployment approaches for Railway:

### Approach 1: Docker-based Deployment (Recommended)

We've set up Docker-based deployment with these files:
- `Dockerfile`: Defines the container environment with all necessary dependencies
- `railway.toml`: Configures Railway to use the Dockerfile for deployment
- `requirements.txt`: Lists all Python dependencies

The Docker-based approach provides a more controlled environment and ensures all system dependencies for OpenCV are properly installed.

### Approach 2: Nixpacks-based Deployment (Alternative)

Alternatively, you can use the Nixpacks approach with these files:
- `Procfile`: Tells Railway how to run your application
- `runtime.txt`: Specifies the Python version
- `nixpacks.toml`: Configures system dependencies for OpenCV and other libraries

However, the Docker approach is more reliable for this application due to the complex system dependencies required by OpenCV.

## Step 2: Set Up Supabase for File Storage

Since Railway has an ephemeral filesystem (files don't persist between deployments), you need to use Supabase for file storage in production:

1. Create a Supabase project at [supabase.com](https://supabase.com/)
2. Create a storage bucket named "videos" in your Supabase project
3. Set the bucket's privacy to "public" or configure appropriate policies
4. Note your Supabase URL and API key for the next step

## Step 3: Deploy to Railway

### Option 1: Deploy via Railway Dashboard (Recommended for beginners)

1. Go to [Railway](https://railway.app/) and log in
2. Click "New Project" > "Deploy from GitHub repo"
3. Select your repository (ai-editor-backend)
4. Railway will automatically detect your Python project and start the deployment

### Option 2: Deploy via Railway CLI

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Create a new project: `railway init`
4. Deploy: `railway up`

## Step 4: Configure Environment Variables

In the Railway dashboard:
1. Go to your project
2. Click on "Variables"
3. Add the following environment variables:
   - `API_HOST`: 0.0.0.0
   - `SUPABASE_URL`: Your Supabase URL
   - `SUPABASE_KEY`: Your Supabase API key
   - `DEBUG`: False (for production)
   - Any other variables from your .env file as needed

Railway automatically sets the `PORT` variable, which our Procfile uses.

## Step 5: Verify Deployment

1. Once deployed, Railway will provide you with a URL (e.g., https://your-app-name.up.railway.app)
2. Visit `https://your-app-name.up.railway.app/docs` to see the Swagger UI documentation
3. Test your API endpoints to ensure everything is working correctly

## Step 6: Connect Your Frontend

Update your frontend configuration to use the new backend URL:
1. In your frontend environment variables, update the API endpoint to your Railway URL
2. Deploy your frontend to a service like Vercel, Netlify, or Railway

## Troubleshooting

- **Deployment Fails**: Check the logs in Railway dashboard for specific errors
- **Storage Issues**: Verify your Supabase credentials and bucket permissions
- **Memory/CPU Limits**: If you encounter resource limits, consider upgrading your Railway plan
- **OpenCV Errors**: If you see errors like `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`, this is likely due to missing system dependencies. If using the Docker approach, you may need to add additional libraries to the Dockerfile. If using the Nixpacks approach, update the `nixpacks.toml` file.
- **Docker Build Errors**: If you encounter Docker build errors, check that all system dependencies are correctly specified in the Dockerfile. You may need to add additional packages depending on your specific OpenCV usage.
- **Railway Builder Selection**: If Railway is not using your Dockerfile, make sure the `railway.toml` file is correctly configured with `builder = "DOCKERFILE"`.

## Important Notes for Production

1. **File Storage**: The `uploads/` and `outputs/` directories in your code are temporary on Railway. All persistent files should be stored in Supabase.

2. **Environment Variables**: Never commit sensitive information like API keys to your repository.

3. **Scaling**: Railway can automatically scale your application, but be mindful of resource usage, especially with video processing.

4. **Monitoring**: Use Railway's built-in monitoring to keep track of your application's performance.
