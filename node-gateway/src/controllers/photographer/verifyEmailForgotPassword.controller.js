// Verify Email for Forgot Password: Verify the verification code
import { verifyPhotographerEmail } from "../../services/photographer.service.js";

export const verifyEmailForgotPassword = async (req, res) => {
  try {
    const { photographerId, verificationCode } = req.body;

    // Validate fields
    if (!photographerId || !verificationCode) {
      return res.status(400).json({ error: "photographerId and verificationCode are required" });
    }

    // Verify code
    const photographer = await verifyPhotographerEmail(photographerId, verificationCode);

    return res.status(200).json({
      message: "Verification code verified successfully",
      photographerId: photographer._id,
      email: photographer.email
    });
  } catch (err) {
    const msg = err.message.toLowerCase();
    if (msg.includes("invalid") || msg.includes("expired")) {
      return res.status(401).json({ error: err.message });
    }
    return res.status(500).json({ error: "Verification failed", details: err.message });
  }
};
