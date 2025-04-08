import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

# Import our custom modules
from app.video_processing import (
    analyze_reference_video,
    process_video,
    extract_scenes,
    apply_color_grading
)
from app.storage import upload_to_storage, get_download_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Video Editor API",
    description="API for AI-powered video editing",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for file storage
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files directory for serving processed videos
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Pydantic models for request/response
class ProcessingResult(BaseModel):
    video_url: str
    thumbnail_url: str
    duration: float

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0
    result: ProcessingResult = None

# In-memory storage for job status (in a real app, use a database)
processing_jobs = {}

@app.get("/")
async def root():
    return {"message": "AI Video Editor API"}

@app.post("/api/upload-reference", status_code=201)
async def upload_reference_video(reference_video: UploadFile = File(...)):
    """
    Upload a reference video for style extraction
    """
    try:
        # Generate a unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(reference_video.filename)[1]
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(reference_video.file, buffer)
        
        logger.info(f"Reference video saved: {file_path}")
        
        return {
            "file_id": file_id,
            "filename": filename,
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error uploading reference video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-media", status_code=201)
async def upload_media_files(media_files: List[UploadFile] = File(...)):
    """
    Upload media files (videos or images) for processing
    """
    try:
        uploaded_files = []
        
        for media_file in media_files:
            # Generate a unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(media_file.filename)[1]
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(media_file.file, buffer)
            
            uploaded_files.append({
                "file_id": file_id,
                "filename": filename,
                "file_path": file_path,
                "content_type": media_file.content_type
            })
        
        logger.info(f"Uploaded {len(uploaded_files)} media files")
        
        return {
            "files": uploaded_files
        }
    except Exception as e:
        logger.error(f"Error uploading media files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_task(
    job_id: str,
    reference_video_path: str,
    media_files: List[dict]
):
    """
    Background task for video processing
    """
    try:
        # Update job status
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="analyzing",
            progress=10
        )
        
        # Step 1: Analyze reference video
        logger.info(f"Analyzing reference video: {reference_video_path}")
        style_params = analyze_reference_video(reference_video_path)
        
        # Update job status
        processing_jobs[job_id].status = "extracting_scenes"
        processing_jobs[job_id].progress = 30
        
        # Step 2: Extract scenes from media files
        logger.info("Extracting scenes from media files")
        media_paths = [file["file_path"] for file in media_files]
        scenes = extract_scenes(media_paths)
        
        # Update job status
        processing_jobs[job_id].status = "applying_color_grading"
        processing_jobs[job_id].progress = 50
        
        # Step 3: Apply color grading
        logger.info("Applying color grading")
        graded_scenes = apply_color_grading(scenes, style_params)
        
        # Update job status
        processing_jobs[job_id].status = "generating_video"
        processing_jobs[job_id].progress = 70
        
        # Step 4: Process video with the extracted style
        logger.info("Generating final video")
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        thumbnail_path = os.path.join(OUTPUT_DIR, f"{job_id}_thumbnail.jpg")
        duration = process_video(graded_scenes, style_params, output_path, thumbnail_path)
        
        # Update job status
        processing_jobs[job_id].status = "uploading"
        processing_jobs[job_id].progress = 90
        
        # Step 5: Upload to storage (optional for MVP)
        # In a real implementation, upload to Supabase or other storage
        # video_url = upload_to_storage(output_path)
        # thumbnail_url = upload_to_storage(thumbnail_path)
        
        # For MVP, we'll just use local paths
        video_url = f"/static/{job_id}.mp4"
        thumbnail_url = f"/static/{job_id}_thumbnail.jpg"
        
        # Update job status to completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].result = ProcessingResult(
            video_url=video_url,
            thumbnail_url=thumbnail_url,
            duration=duration
        )
        
        logger.info(f"Video processing completed: {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="failed",
            progress=0
        )

@app.post("/api/process-video", status_code=202)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    reference_video_id: str,
    media_file_ids: List[str]
):
    """
    Process video with AI using the reference video style
    """
    try:
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Get file paths
        reference_video_path = os.path.join(UPLOAD_DIR, f"{reference_video_id}.mp4")
        
        media_files = []
        for file_id in media_file_ids:
            # In a real implementation, you would look up the file details from a database
            # For MVP, we'll assume the extension based on the file ID
            file_path = None
            for ext in [".mp4", ".jpg", ".png", ".jpeg"]:
                test_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
                if os.path.exists(test_path):
                    file_path = test_path
                    break
            
            if file_path:
                media_files.append({
                    "file_id": file_id,
                    "file_path": file_path
                })
        
        # Initialize job status
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="queued",
            progress=0
        )
        
        # Start background processing task
        background_tasks.add_task(
            process_video_task,
            job_id,
            reference_video_path,
            media_files
        )
        
        return {
            "job_id": job_id,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the processed video
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Video processing not completed")
    
    video_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"ai-edited-video-{job_id}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
