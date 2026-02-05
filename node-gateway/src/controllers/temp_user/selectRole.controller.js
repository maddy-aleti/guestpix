// Select Role: user only in this flow
export const selectRole = async (req, res) => {
  try {
    const { role } = req.body;

    if (!role) {
      return res.status(400).json({ error: "Role is required" });
    }
    if (!["user", "photographer"].includes(role)) {
      return res.status(400).json({ error: "Invalid role. Use 'user' or 'photographer'" });
    }
    if (role !== "user") {
      return res.status(400).json({
        error: "Only 'user' flow is supported here. Photographer flow is separate."
      });
    }

    return res.json({
      message: "Role selected. Proceed to sign in (email).",
      role
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to select role", details: err.message });
  }
};