// src/controllers/photographer/generateUserQR.controller.js
import { generateQRCode } from "../../services/photographer.service.js";

export const generateUserQRController = async (req, res) => {
  try {
    const { projectId, photographerId } = req.body;
    // TODO: After middleware is implemented, get photographerId from req.photographerId

    // Validate required fields
    if (!projectId || !photographerId) {
      return res.status(400).json({
        success: false,
        message: "projectId and photographerId are required"
      });
    }

    // Generate QR code for user (24 hours validity)
    const qrData = await generateQRCode({
      projectId,
      photographerId,
      qrType: "user",
      validityHours: 24
    });

    return res.status(201).json({
      success: true,
      message: "User QR code generated successfully (valid for 24 hours)",
      data: {
        qrCodeId: qrData._id,
        qrCode: qrData.qrCode,
        qrImageUrl: qrData.qrImageUrl,
        qrType: qrData.qrType,
        expiresAt: qrData.expiresAt,
        projectId: qrData.project,
        createdAt: qrData.createdAt
      }
    });
  } catch (err) {
    console.error("Error in generateUserQR:", err);
    return res.status(500).json({
      success: false,
      message: err.message || "Failed to generate user QR code"
    });
  }
};