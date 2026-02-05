// Sign Up: Create photographer with name, email, password
import bcryptjs from "bcryptjs";
import Photographer from "../../models/photographer.model.js";

export const signup = async (req, res) => {
  try {
    const { name, email, password, confirmPassword } = req.body;

    // Validate fields
    if (!name || !email || !password || !confirmPassword) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: "Invalid email format" });
    }

    // Check password match
    if (password !== confirmPassword) {
      return res.status(400).json({ error: "Passwords do not match" });
    }

    // Check password length
    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters" });
    }

    // Check if email already exists
    const existingPhotographer = await Photographer.findOne({ email: email.toLowerCase() });
    if (existingPhotographer) {
      return res.status(409).json({ error: "Email already registered" });
    }

    // Hash password
    const hashedPassword = await bcryptjs.hash(password, 10);

    // Create photographer
    const photographer = await Photographer.create({
      name,
      email: email.toLowerCase(),
      password: hashedPassword,
      role: "photographer"
    });

    return res.status(201).json({
      message: "Account created successfully",
      photographerId: photographer._id,
      email: photographer.email,
      name: photographer.name
    });
  } catch (err) {
    return res.status(500).json({ error: "Sign up failed", details: err.message });
  }
};
