// src/services/photographer.service.js
import Photographer from "../models/photographer.model.js";
import Project from "../models/project.model.js";
import QRCode from "../models/qrCode.model.js";
import crypto from "crypto";
import qrcode from "qrcode";

export const generateVerificationCode = () => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

export const getCodeExpiry = () => {
  return new Date(Date.now() + 10 * 60 * 1000); // 10 minutes
};

export const createPhotographer = async (email) => {
  try {
    // Check if email already exists in Photographer
    const existingPhotographer = await Photographer.findOne({ email: email.toLowerCase() });
    if (existingPhotographer) {
      throw new Error("Email already registered");
    }

    const photographer = await Photographer.create({
      email: email.toLowerCase(),
      role: "photographer",
      isVerified: false
    });

    return photographer;
  } catch (err) {
    throw err;
  }
};

export const updateVerificationCode = async (photographerId, verificationCode) => {
  try {
    const photographer = await Photographer.findByIdAndUpdate(
      photographerId,
      {
        verificationCode,
        verificationCodeExpiry: getCodeExpiry()
      },
      { new: true }
    );
    return photographer;
  } catch (err) {
    throw err;
  }
};

export const verifyPhotographerEmail = async (photographerId, verificationCode) => {
  try {
    const photographer = await Photographer.findById(photographerId);

    if (!photographer) {
      throw new Error("Photographer not found");
    }

    if (photographer.verificationCode !== verificationCode) {
      throw new Error("Invalid verification code");
    }

    if (new Date() > photographer.verificationCodeExpiry) {
      throw new Error("Verification code expired");
    }

    const updatedPhotographer = await Photographer.findByIdAndUpdate(
      photographerId,
      {
        isVerified: true,
        verificationCode: null,
        verificationCodeExpiry: null
      },
      { new: true }
    );

    return updatedPhotographer;
  } catch (err) {
    throw err;
  }
};

export const deletePhotographer = async (photographerId) => {
  try {
    await Photographer.findByIdAndDelete(photographerId);
  } catch (err) {
    throw err;
  }
};

export const createProject = async ({ projectName, eventOwnerName, eventName, date, photographerId }) => {
  try {
    // Verify photographer exists
    const photographer = await Photographer.findById(photographerId);
    if (!photographer) {
      throw new Error("Photographer not found");
    }

    const project = await Project.create({
      projectName,
      eventOwnerName,
      eventName,
      date,
      photographer: photographerId
    });

    return project;
  } catch (err) {
    throw err;
  }
};

export const generateQRCode = async ({ projectId, photographerId, qrType, validityHours }) => {
  try {
    // Verify project exists and belongs to photographer
    const project = await Project.findOne({
      _id: projectId,
      photographer: photographerId
    });

    if (!project) {
      throw new Error("Project not found or unauthorized");
    }

    // Generate unique QR code string
    const qrCodeString = crypto.randomBytes(32).toString("hex");

    // Calculate expiry time
    const expiresAt = new Date(Date.now() + validityHours * 60 * 60 * 1000);

    // Check if active QR code already exists for this project and type
    const existingQR = await QRCode.findOne({
      project: projectId,
      qrType: qrType,
      isActive: true,
      expiresAt: { $gt: new Date() }
    });

    if (existingQR) {
      // Return existing QR code if still valid
      return existingQR;
    }

    // Generate QR code image as data URL
    const qrImageUrl = await qrcode.toDataURL(qrCodeString, {
      errorCorrectionLevel: 'H',
      type: 'image/png',
      width: 300,
      margin: 2
    });

    // Create new QR code
    const qrData = await QRCode.create({
      project: projectId,
      photographer: photographerId,
      qrType,
      qrCode: qrCodeString,
      qrImageUrl,
      expiresAt
    });

    return qrData;
  } catch (err) {
    throw err;
  }
};