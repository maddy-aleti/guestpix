// Sign In: Authenticate photographer with email and password
import bcryptjs from "bcryptjs";
import Photographer from "../../models/photographer.model.js";

export const signin = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate fields
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Find photographer
    const photographer = await Photographer.findOne({ email: email.toLowerCase() });
    if (!photographer) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    // Compare password
    const isPasswordValid = await bcryptjs.compare(password, photographer.password);
    if (!isPasswordValid) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    return res.status(200).json({
      message: "Sign in successful",
      photographerId: photographer._id,
      email: photographer.email,
      name: photographer.name
    });
  } catch (err) {
    return res.status(500).json({ error: "Sign in failed", details: err.message });
  }
};
