import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from skimage import color, exposure

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_reference_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze reference video to extract editing style parameters
    
    Parameters:
    - video_path: Path to the reference video file
    
    Returns:
    - Dictionary containing style parameters
    """
    logger.info(f"Analyzing reference video: {video_path}")
    
    try:
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract basic parameters
        duration = video.duration
        fps = video.fps
        width, height = video.size
        
        # Calculate average shot length
        scenes = detect_scenes_opencv(video_path)
        if len(scenes) > 1:
            shot_lengths = [scenes[i][1] - scenes[i][0] for i in range(len(scenes))]
            avg_shot_length = sum(shot_lengths) / len(shot_lengths)
        else:
            avg_shot_length = duration
        
        # Extract color profile
        color_profile = extract_color_profile(video)
        
        # Extract transition types (simplified for MVP)
        # In a real implementation, this would use ML to detect transition types
        transitions = {
            "cut": 0.7,  # 70% probability of using cuts
            "fade": 0.2,  # 20% probability of using fades
            "dissolve": 0.1  # 10% probability of using dissolves
        }
        
        # Extract composition parameters (simplified for MVP)
        composition = {
            "rule_of_thirds": True,
            "symmetry": False,
            "leading_lines": False
        }
        
        # Create style parameters dictionary
        style_params = {
            "duration": duration,
            "fps": fps,
            "resolution": (width, height),
            "avg_shot_length": avg_shot_length,
            "color_profile": color_profile,
            "transitions": transitions,
            "composition": composition
        }
        
        logger.info(f"Extracted style parameters: {style_params}")
        return style_params
        
    except Exception as e:
        logger.error(f"Error analyzing reference video: {str(e)}")
        # Return default parameters if analysis fails
        return {
            "duration": 30,
            "fps": 30,
            "resolution": (1280, 720),
            "avg_shot_length": 2.5,
            "color_profile": {
                "contrast": 1.0,
                "brightness": 1.0,
                "saturation": 1.0,
                "temperature": 0
            },
            "transitions": {
                "cut": 0.8,
                "fade": 0.1,
                "dissolve": 0.1
            },
            "composition": {
                "rule_of_thirds": True,
                "symmetry": False,
                "leading_lines": False
            }
        }

def detect_scenes_opencv(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Detect scene changes in a video using OpenCV
    
    Parameters:
    - video_path: Path to the video file
    - threshold: Threshold for scene change detection (higher = fewer scenes)
    
    Returns:
    - List of tuples containing (start_time, end_time) for each scene
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or frame_count <= 0:
            logger.warning(f"Invalid video properties: fps={fps}, frames={frame_count}")
            # Fallback to using moviepy to get properties
            video = VideoFileClip(video_path)
            fps = video.fps
            frame_count = int(video.duration * fps)
            video.close()
        
        # Initialize variables
        prev_frame = None
        scene_changes = []
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if prev_frame is not None:
                # Calculate absolute difference between current and previous frame
                frame_diff = cv2.absdiff(gray, prev_frame)
                
                # Calculate mean difference
                mean_diff = np.mean(frame_diff)
                
                # If difference is above threshold, mark as scene change
                if mean_diff > threshold:
                    scene_changes.append(frame_idx)
            
            # Update previous frame
            prev_frame = gray
            frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Convert frame indices to time ranges
        scenes = []
        if len(scene_changes) > 0:
            # Add first scene (from start to first scene change)
            scenes.append((0, scene_changes[0] / fps))
            
            # Add middle scenes
            for i in range(len(scene_changes) - 1):
                scenes.append((scene_changes[i] / fps, scene_changes[i + 1] / fps))
            
            # Add last scene (from last scene change to end)
            scenes.append((scene_changes[-1] / fps, frame_count / fps))
        else:
            # If no scene changes detected, treat the whole video as one scene
            scenes.append((0, frame_count / fps))
        
        return scenes
    
    except Exception as e:
        logger.error(f"Error detecting scenes with OpenCV: {str(e)}")
        # Return a single scene covering the entire video
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            return [(0, duration)]
        except:
            return [(0, 30)]  # Default 30 seconds if everything fails

def extract_color_profile(video: VideoFileClip) -> Dict[str, float]:
    """
    Extract color profile from a video
    
    Parameters:
    - video: VideoFileClip object
    
    Returns:
    - Dictionary containing color profile parameters
    """
    try:
        # Sample frames from the video
        num_samples = min(10, int(video.duration))
        sample_times = np.linspace(0, video.duration, num_samples, endpoint=False)
        
        # Initialize color profile parameters
        contrast_values = []
        brightness_values = []
        saturation_values = []
        temperature_values = []
        
        for t in sample_times:
            # Get frame at time t
            frame = video.get_frame(t)
            
            # Convert to LAB color space for better analysis
            lab_frame = color.rgb2lab(frame / 255.0)
            
            # Extract L channel (lightness)
            l_channel = lab_frame[:, :, 0]
            
            # Calculate contrast (standard deviation of lightness)
            contrast = np.std(l_channel)
            contrast_values.append(contrast)
            
            # Calculate brightness (mean lightness)
            brightness = np.mean(l_channel)
            brightness_values.append(brightness)
            
            # Calculate saturation (mean of a and b channels)
            saturation = np.mean(np.sqrt(lab_frame[:, :, 1]**2 + lab_frame[:, :, 2]**2))
            saturation_values.append(saturation)
            
            # Calculate color temperature (blue-yellow balance)
            # Positive values indicate warmer (yellow), negative values indicate cooler (blue)
            temperature = np.mean(lab_frame[:, :, 2])
            temperature_values.append(temperature)
        
        # Average the values
        avg_contrast = np.mean(contrast_values)
        avg_brightness = np.mean(brightness_values)
        avg_saturation = np.mean(saturation_values)
        avg_temperature = np.mean(temperature_values)
        
        # Normalize values to a more intuitive range
        normalized_contrast = avg_contrast / 20.0  # Typical range is 0-40
        normalized_brightness = avg_brightness / 100.0  # Range is 0-100
        normalized_saturation = avg_saturation / 30.0  # Typical range is 0-60
        normalized_temperature = avg_temperature / 20.0  # Typical range is -20 to 20
        
        return {
            "contrast": normalized_contrast,
            "brightness": normalized_brightness,
            "saturation": normalized_saturation,
            "temperature": normalized_temperature
        }
    
    except Exception as e:
        logger.error(f"Error extracting color profile: {str(e)}")
        return {
            "contrast": 1.0,
            "brightness": 1.0,
            "saturation": 1.0,
            "temperature": 0.0
        }

def extract_scenes(media_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Extract scenes from media files
    
    Parameters:
    - media_paths: List of paths to media files (videos or images)
    
    Returns:
    - List of scene dictionaries
    """
    scenes = []
    
    for path in media_paths:
        try:
            # Check if file is video or image
            if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                # Process video file
                video_scenes = detect_scenes_opencv(path)
                
                for i, (start_time, end_time) in enumerate(video_scenes):
                    scenes.append({
                        "type": "video",
                        "path": path,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time
                    })
            
            elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Process image file (treat as a 3-second scene)
                scenes.append({
                    "type": "image",
                    "path": path,
                    "duration": 3.0
                })
        
        except Exception as e:
            logger.error(f"Error processing media file {path}: {str(e)}")
    
    return scenes

def apply_color_grading(scenes: List[Dict[str, Any]], style_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply color grading to scenes based on reference style
    
    Parameters:
    - scenes: List of scene dictionaries
    - style_params: Style parameters extracted from reference video
    
    Returns:
    - List of graded scene dictionaries
    """
    color_profile = style_params.get("color_profile", {
        "contrast": 1.0,
        "brightness": 1.0,
        "saturation": 1.0,
        "temperature": 0.0
    })
    
    graded_scenes = []
    
    for scene in scenes:
        # Create a copy of the scene
        graded_scene = scene.copy()
        
        # Add color grading parameters
        graded_scene["color_grading"] = {
            "contrast": color_profile.get("contrast", 1.0),
            "brightness": color_profile.get("brightness", 1.0),
            "saturation": color_profile.get("saturation", 1.0),
            "temperature": color_profile.get("temperature", 0.0)
        }
        
        graded_scenes.append(graded_scene)
    
    return graded_scenes

def apply_color_grade_to_frame(frame, color_grading):
    """
    Apply color grading to a single frame
    
    Parameters:
    - frame: NumPy array representing the frame
    - color_grading: Color grading parameters
    
    Returns:
    - Graded frame
    """
    # Convert to float for processing
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply contrast
    contrast = color_grading.get("contrast", 1.0)
    frame_float = exposure.adjust_gamma(frame_float, 1.0 / contrast)
    
    # Apply brightness
    brightness = color_grading.get("brightness", 1.0)
    frame_float = frame_float * brightness
    
    # Apply saturation
    saturation = color_grading.get("saturation", 1.0)
    hsv = cv2.cvtColor(frame_float, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Apply temperature (color balance)
    temperature = color_grading.get("temperature", 0.0)
    if temperature > 0:
        # Warmer (add yellow/red)
        frame_float[:, :, 0] = frame_float[:, :, 0] * (1 + 0.1 * temperature)  # R
        frame_float[:, :, 2] = frame_float[:, :, 2] * (1 - 0.1 * temperature)  # B
    else:
        # Cooler (add blue)
        frame_float[:, :, 0] = frame_float[:, :, 0] * (1 + 0.1 * temperature)  # R
        frame_float[:, :, 2] = frame_float[:, :, 2] * (1 - 0.1 * temperature)  # B
    
    # Clip values to valid range
    frame_float = np.clip(frame_float, 0, 1)
    
    # Convert back to uint8
    return (frame_float * 255).astype(np.uint8)

def process_video(
    scenes: List[Dict[str, Any]], 
    style_params: Dict[str, Any], 
    output_path: str,
    thumbnail_path: str
) -> float:
    """
    Process video with the extracted style
    
    Parameters:
    - scenes: List of scene dictionaries
    - style_params: Style parameters extracted from reference video
    - output_path: Path to save the output video
    - thumbnail_path: Path to save the thumbnail image
    
    Returns:
    - Duration of the processed video
    """
    try:
        # Get target parameters
        target_fps = style_params.get("fps", 30)
        target_resolution = style_params.get("resolution", (1280, 720))
        transitions = style_params.get("transitions", {"cut": 0.8, "fade": 0.1, "dissolve": 0.1})
        
        # Process each scene
        clips = []
        
        for scene in scenes:
            scene_type = scene.get("type")
            color_grading = scene.get("color_grading", {})
            
            if scene_type == "video":
                # Load video clip
                video = VideoFileClip(scene.get("path"))
                
                # Trim to scene boundaries
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", video.duration)
                video = video.subclip(start_time, end_time)
                
                # Apply color grading
                video = video.fl_image(lambda frame: apply_color_grade_to_frame(frame, color_grading))
                
                clips.append(video)
                
            elif scene_type == "image":
                # Load image clip
                image = ImageClip(scene.get("path"))
                
                # Set duration
                duration = scene.get("duration", 3.0)
                image = image.set_duration(duration)
                
                # Apply color grading
                image = image.fl_image(lambda frame: apply_color_grade_to_frame(frame, color_grading))
                
                clips.append(image)
        
        # Apply transitions between clips
        final_clips = []
        
        for i, clip in enumerate(clips):
            final_clips.append(clip)
            
            # Add transition to next clip (except for the last clip)
            if i < len(clips) - 1:
                # Choose transition type based on probabilities
                transition_type = np.random.choice(
                    ["cut", "fade", "dissolve"], 
                    p=[transitions.get("cut", 0.8), transitions.get("fade", 0.1), transitions.get("dissolve", 0.1)]
                )
                
                # Apply transition (simplified for MVP)
                # In a real implementation, this would be more sophisticated
                if transition_type == "fade" or transition_type == "dissolve":
                    # Add a short crossfade
                    clips[i] = clips[i].crossfadeout(0.5)
                    clips[i+1] = clips[i+1].crossfadein(0.5)
        
        # Concatenate clips
        final_video = concatenate_videoclips(clips)
        
        # Resize to target resolution
        final_video = final_video.resize(target_resolution)
        
        # Set fps
        final_video = final_video.set_fps(target_fps)
        
        # Write output video
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4
        )
        
        # Generate thumbnail
        thumbnail_time = min(1.0, final_video.duration / 2)
        thumbnail = final_video.get_frame(thumbnail_time)
        cv2.imwrite(thumbnail_path, cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR))
        
        return final_video.duration
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        
        # Create a simple error video
        try:
            # Create a blank video with error message
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image
            img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Add error message
            draw.text((640, 360), "Error processing video", fill=(255, 255, 255), anchor="mm")
            
            # Save as video
            img_path = output_path.replace(".mp4", "_error.jpg")
            img.save(img_path)
            
            # Create video clip
            error_clip = ImageClip(img_path).set_duration(5)
            error_clip.write_videofile(
                output_path,
                codec="libx264",
                fps=30
            )
            
            # Save thumbnail
            img.save(thumbnail_path)
            
            # Clean up
            if os.path.exists(img_path):
                os.remove(img_path)
            
            return 5.0
        
        except Exception as inner_e:
            logger.error(f"Error creating error video: {str(inner_e)}")
            return 0.0
