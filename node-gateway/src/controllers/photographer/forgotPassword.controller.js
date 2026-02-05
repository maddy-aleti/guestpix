// Forgot Password: Send verification code to email
import Photographer from "../../models/photographer.model.js";
import { generateVerificationCode, updateVerificationCode } from "../../services/photographer.service.js";
import { sendVerificationEmail } from "../../services/email.service.js";

export const forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    // Validate email
    if (!email) {
      return res.status(400).json({ error: "Email is required" });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: "Invalid email format" });
    }

    // Find photographer
    const photographer = await Photographer.findOne({ email: email.toLowerCase() });
    if (!photographer) {
      return res.status(404).json({ error: "Photographer not found" });
    }

    // Generate verification code
    const verificationCode = generateVerificationCode();
    await updateVerificationCode(photographer._id, verificationCode);
    
    // Send email with verification code
    await sendVerificationEmail(email, verificationCode);

    return res.status(200).json({
      message: "Verification code sent to your email",
      photographerId: photographer._id,
      email: photographer.email
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to send verification code", details: err.message });
  }
};
