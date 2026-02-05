// Reset Password: Reset password after verification
import bcryptjs from "bcryptjs";
import Photographer from "../../models/photographer.model.js";

export const resetPassword = async (req, res) => {
  try {
    const { photographerId, newPassword, confirmPassword } = req.body;

    // Validate fields
    if (!photographerId || !newPassword || !confirmPassword) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Check password match
    if (newPassword !== confirmPassword) {
      return res.status(400).json({ error: "Passwords do not match" });
    }

    // Check password length
    if (newPassword.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters" });
    }

    // Find photographer
    const photographer = await Photographer.findById(photographerId);
    if (!photographer) {
      return res.status(404).json({ error: "Photographer not found" });
    }

    // Hash new password
    const hashedPassword = await bcryptjs.hash(newPassword, 10);

    // Update password and clear verification code
    photographer.password = hashedPassword;
    photographer.verificationCode = null;
    photographer.verificationCodeExpiry = null;
    await photographer.save();

    return res.status(200).json({
      message: "Password reset successfully",
      photographerId: photographer._id,
      email: photographer.email
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to reset password", details: err.message });
  }
};
