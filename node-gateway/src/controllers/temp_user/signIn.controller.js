// Sign In: create temp user + email verification code
import { createTempUser, generateVerificationCode, updateVerificationCode } from "../../services/temp_user.service.js";
import { sendVerificationEmail } from "../../services/email.service.js";

export const signIn = async (req, res) => {
  try {
    const { email, role } = req.body;

    if (!email || !role) {
      return res.status(400).json({ error: "Email and role are required" });
    }
    if (role !== "user") {
      return res.status(400).json({ error: "Invalid role for this flow" });
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: "Invalid email format" });
    }

    const tempUser = await createTempUser(email, role);

    const verificationCode = generateVerificationCode();
    await updateVerificationCode(tempUser._id, verificationCode);
    await sendVerificationEmail(email, verificationCode);

    return res.status(201).json({
      message: "Verification code sent to your email",
      tempUserId: tempUser._id,
      email: tempUser.email
    });
  } catch (err) {
    if (err.message.includes("Email already registered")) {
      return res.status(409).json({ error: err.message });
    }
    return res.status(500).json({ error: "Sign in failed", details: err.message });
  }
};