// src/controllers/photographer/generateClientQR.controller.js
import { generateQRCode } from "../../services/photographer.service.js";

export const generateClientQRController = async (req, res) => {
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

    // Generate QR code for client (30 days validity)
    const qrData = await generateQRCode({
      projectId,
      photographerId,
      qrType: "client",
      validityHours: 720 // 30 days = 720 hours
    });

    return res.status(201).json({
      success: true,
      message: "Client QR code generated successfully (valid for 30 days)",
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
    console.error("Error in generateClientQR:", err);
    return res.status(500).json({
      success: false,
      message: err.message || "Failed to generate client QR code"
    });
  }
};