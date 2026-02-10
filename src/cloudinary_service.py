"""
Cloudinary Cloud Storage Service
Handles uploading and retrieving images from Cloudinary.
"""

import cloudinary
import cloudinary.uploader
import cloudinary.api
import yaml
from pathlib import Path
from typing import Optional, Dict


class CloudinaryService:
    """Service for managing image uploads to Cloudinary."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Cloudinary with credentials from config."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        cloudinary_config = self.config.get('cloudinary', {})
        
        # Configure cloudinary
        cloudinary.config(
            cloud_name=cloudinary_config.get('cloud_name'),
            api_key=cloudinary_config.get('api_key'),
            api_secret=cloudinary_config.get('api_secret')
        )
        
        self.cloud_name = cloudinary_config.get('cloud_name')
        self.api_key = cloudinary_config.get('api_key')
        self.api_secret = cloudinary_config.get('api_secret')
        self.use_cloud_storage = cloudinary_config.get('use_cloud_storage', False)
        self.raw_photos_folder = cloudinary_config.get('raw_photos_folder', 'face_recognition/raw_photos')
        self.cropped_faces_folder = cloudinary_config.get('cropped_faces_folder', 'face_recognition/cropped_faces')
        
        # Validate configuration
        if self.use_cloud_storage:
            if not all([self.cloud_name, self.api_key, self.api_secret]):
                raise ValueError("Cloudinary credentials (cloud_name, api_key, api_secret) are required when use_cloud_storage is enabled")
            print("✓ Cloudinary service initialized successfully")
        else:
            print("ℹ Cloud storage is disabled. Using local storage.")
    
    def upload_photo(self, file_path: str, public_id: Optional[str] = None, folder: str = None) -> Dict:
        """
        Upload a photo to Cloudinary.
        
        Args:
            file_path: Path to local image file
            public_id: Optional public ID for the image (without folder prefix)
            folder: Optional Cloudinary folder path
            
        Returns:
            Dictionary with upload result including secure_url
        """
        if not self.use_cloud_storage:
            return {"error": "Cloud storage is disabled"}
        
        try:
            # Use the appropriate folder
            if folder is None:
                folder = self.raw_photos_folder
            
            # Prepare upload options
            upload_options = {
                'folder': folder,
                'resource_type': 'auto',
                'overwrite': False
            }
            
            # If public_id is provided, it should be just the filename/relative path
            # The folder is handled separately
            if public_id:
                upload_options['public_id'] = public_id
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                file_path,
                **upload_options
            )
            
            print(f"✓ Uploaded to Cloudinary: {result.get('secure_url')}")
            return result
            
        except Exception as e:
            print(f"✗ Failed to upload {file_path} to Cloudinary: {e}")
            raise
    
    def upload_cropped_face(self, file_path: str, public_id: Optional[str] = None) -> Dict:
        """
        Upload a cropped face image to Cloudinary.
        
        Args:
            file_path: Path to cropped face file
            public_id: Optional public ID for the image
            
        Returns:
            Dictionary with upload result
        """
        return self.upload_photo(file_path, public_id, folder=self.cropped_faces_folder)
    
    def delete_file(self, public_id: str) -> Dict:
        """
        Delete a file from Cloudinary.
        
        Args:
            public_id: Public ID of the file to delete
            
        Returns:
            Deletion result
        """
        try:
            result = cloudinary.uploader.destroy(public_id)
            print(f"✓ Deleted from Cloudinary: {public_id}")
            return result
        except Exception as e:
            print(f"✗ Failed to delete {public_id} from Cloudinary: {e}")
            raise
    
    def get_secure_url(self, public_id: str) -> str:
        """
        Get secure HTTP URL for an uploaded image.
        
        Args:
            public_id: Public ID of the image
            
        Returns:
            Secure HTTPS URL
        """
        from cloudinary.utils import cloudinary_url
        url, _ = cloudinary_url(public_id, secure=True)
        return url
