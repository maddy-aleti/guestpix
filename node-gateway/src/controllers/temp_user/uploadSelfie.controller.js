import multer from "multer";
import path from "path";
import fs from "fs";
import TempUser from "../../models/tempUser.model.js";

// Create upload directory if it doesn't exist
const uploadDir = path.join(process.cwd(), "../../data/user_selfies");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Generate unique filename: email_timestamp.extension
    const uniqueName = `${Date.now()}_${file.originalname}`;
    cb(null, uniqueName);
  }
});

// File filter - only accept images
const fileFilter = (req, file, cb) => {
  const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error("Only JPEG, JPG, and PNG images are allowed"), false);
  }
};

// Multer upload instance
export const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Controller to handle selfie upload
export const uploadSelfie = async (req, res) => {
  try {
    const { email } = req.body;

    // Validate email
    if (!email) {
      return res.status(400).json({
        success: false,
        message: "Email is required"
      });
    }

    // Check if file was uploaded
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No file uploaded"
      });
    }

    // Find the temp user
    const tempUser = await TempUser.findOne({ email: email.toLowerCase() });

    if (!tempUser) {
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      return res.status(404).json({
        success: false,
        message: "User not found"
      });
    }

    // Check if user is verified
    if (!tempUser.isVerified) {
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      return res.status(403).json({
        success: false,
        message: "Please verify your email first"
      });
    }

    // Delete old photo if exists
    if (tempUser.photoUrl && fs.existsSync(tempUser.photoUrl)) {
      fs.unlinkSync(tempUser.photoUrl);
    }

    // Update user with photo URL
    tempUser.photoUrl = req.file.path;
    await tempUser.save();

    res.status(200).json({
      success: true,
      message: "Selfie uploaded successfully",
      data: {
        email: tempUser.email,
        photoUrl: tempUser.photoUrl
      }
    });
  } catch (error) {
    console.error("Upload selfie error:", error);
    
    // Clean up uploaded file if error occurs
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      success: false,
      message: "Failed to upload selfie",
      error: error.message
    });
  }
};
