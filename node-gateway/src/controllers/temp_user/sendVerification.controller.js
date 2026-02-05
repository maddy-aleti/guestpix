// Resend only: sign-in already sends the first code
import TempUser from "../../models/tempUser.model.js";
import { generateVerificationCode, updateVerificationCode } from "../../services/temp_user.service.js";
import { sendVerificationEmail } from "../../services/email.service.js";

export const resendVerificationCode = async (req, res) => {
  try {
    const { tempUserId } = req.body;
    if (!tempUserId) {
      return res.status(400).json({ error: "tempUserId is required" });
    }

    const tempUser = await TempUser.findById(tempUserId);
    if (!tempUser) {
      return res.status(404).json({ error: "User not found" });
    }
    if (tempUser.isVerified) {
      return res.status(400).json({ error: "Email already verified" });
    }

    const code = generateVerificationCode();
    await updateVerificationCode(tempUserId, code);
    await sendVerificationEmail(tempUser.email, code);

    return res.json({ message: "New verification code sent" });
  } catch (err) {
    return res.status(500).json({ error: "Failed to resend verification code", details: err.message });
  }
};