import os
import logging
from typing import Optional
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {str(e)}")
else:
    logger.warning("Supabase credentials not found. Storage functionality will be limited.")

def upload_to_storage(file_path: str, bucket_name: str = "videos") -> str:
    """
    Upload a file to Supabase Storage
    
    Parameters:
    - file_path: Path to the file to upload
    - bucket_name: Name of the storage bucket
    
    Returns:
    - URL of the uploaded file
    """
    if not supabase:
        logger.warning("Supabase client not initialized. Using local storage.")
        return file_path
    
    try:
        # Generate a unique file name
        file_name = os.path.basename(file_path)
        unique_name = f"{uuid.uuid4()}_{file_name}"
        
        # Read file
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        # Upload to Supabase
        response = supabase.storage.from_(bucket_name).upload(
            unique_name,
            file_data,
            {"content-type": "video/mp4" if file_path.endswith(".mp4") else "image/jpeg"}
        )
        
        # Get public URL
        file_url = supabase.storage.from_(bucket_name).get_public_url(unique_name)
        
        logger.info(f"File uploaded to Supabase: {file_url}")
        return file_url
    
    except Exception as e:
        logger.error(f"Error uploading to Supabase: {str(e)}")
        return file_path

def get_download_url(file_path: str, bucket_name: str = "videos") -> str:
    """
    Get a download URL for a file in Supabase Storage
    
    Parameters:
    - file_path: Path or name of the file in storage
    - bucket_name: Name of the storage bucket
    
    Returns:
    - Download URL
    """
    if not supabase:
        logger.warning("Supabase client not initialized. Using local path.")
        return file_path
    
    try:
        # If file_path is a local path, extract just the filename
        file_name = os.path.basename(file_path)
        
        # Get download URL
        download_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
        
        logger.info(f"Download URL generated: {download_url}")
        return download_url
    
    except Exception as e:
        logger.error(f"Error generating download URL: {str(e)}")
        return file_path

def create_temporary_link(file_path: str, bucket_name: str = "videos", expires_in: int = 3600) -> str:
    """
    Create a temporary link for a file in Supabase Storage
    
    Parameters:
    - file_path: Path or name of the file in storage
    - bucket_name: Name of the storage bucket
    - expires_in: Expiration time in seconds
    
    Returns:
    - Temporary URL
    """
    if not supabase:
        logger.warning("Supabase client not initialized. Using local path.")
        return file_path
    
    try:
        # If file_path is a local path, extract just the filename
        file_name = os.path.basename(file_path)
        
        # Create signed URL
        signed_url = supabase.storage.from_(bucket_name).create_signed_url(
            file_name,
            expires_in
        )
        
        logger.info(f"Temporary link created: {signed_url}")
        return signed_url
    
    except Exception as e:
        logger.error(f"Error creating temporary link: {str(e)}")
        return file_path

def delete_file(file_path: str, bucket_name: str = "videos") -> bool:
    """
    Delete a file from Supabase Storage
    
    Parameters:
    - file_path: Path or name of the file in storage
    - bucket_name: Name of the storage bucket
    
    Returns:
    - True if successful, False otherwise
    """
    if not supabase:
        logger.warning("Supabase client not initialized. Cannot delete from storage.")
        return False
    
    try:
        # If file_path is a local path, extract just the filename
        file_name = os.path.basename(file_path)
        
        # Delete file
        supabase.storage.from_(bucket_name).remove([file_name])
        
        logger.info(f"File deleted: {file_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return False
