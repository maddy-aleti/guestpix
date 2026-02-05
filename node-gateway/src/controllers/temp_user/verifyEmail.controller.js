import { verifyTempUserEmail } from "../../services/temp_user.service.js";

export const verifyEmail = async (req, res) => {
  try {
    const { tempUserId ,verificationCode } = req.body;

    if (!tempUserId || !verificationCode) {
      return res.status(400).json({ error: "tempUserId and verificationCode are required" });
    }

    const tempUser = await verifyTempUserEmail(tempUserId, verificationCode);

    return res.json({
      message: "Email verified successfully",
      tempUserId: tempUser._id,
      email: tempUser.email,
      role: tempUser.role
    });
  } catch (err) {
    const msg = err.message.toLowerCase();
    if (msg.includes("invalid") || msg.includes("expired")) {
      return res.status(401).json({ error: err.message });
    }
    return res.status(500).json({ error: "Email verification failed", details: err.message });
  }
};