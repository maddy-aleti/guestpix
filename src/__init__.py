"""
Face Recognition Pipeline Package

This package provides a complete face recognition system with three main components:
1. Photo Ingestion & Storage
2. Face Detection
3. Face Embedding & MongoDB Storage
"""

from .photo_ingestion import PhotoIngestion
from .face_detection import FaceDetection
from .face_embedding import FaceEmbedding
from .store_embeddings import MongoDBStorage
from .guest_search import GuestSearchMongoDB

__version__ = "1.0.0"
__author__ = "Face Recognition Team"

__all__ = [
    "PhotoIngestion",
    "FaceDetection", 
    "FaceEmbedding",
    "MongoDBStorage",
    "GuestSearchMongoDB"
] 