import nodemailer from "nodemailer";

const buildTransporter = () =>
  nodemailer.createTransport({
    service: "gmail",
    auth: { user: process.env.EMAIL_USER, pass: process.env.EMAIL_PASSWORD }
  });

export const sendVerificationEmail = async (email, verificationCode) => {
  // Dev fallback if creds missing
  if (!process.env.EMAIL_USER || !process.env.EMAIL_PASSWORD) {
    console.log("\n=== VERIFICATION CODE (DEV) ===");
    console.log("Email:", email);
    console.log("Code :", verificationCode);
    console.log("===============================\n");
    return { success: true };
  }

  try {
    const transporter = buildTransporter();
    await transporter.sendMail({
      from: process.env.EMAIL_USER,
      to: email,
      subject: "Email Verification Code",
      html: `
        <h2>Welcome!</h2>
        <p>Your verification code is:</p>
        <h3 style="color:#007bff; letter-spacing:3px;">${verificationCode}</h3>
        <p>Expires in 10 minutes.</p>
      `
    });
    return { success: true };
  } catch (err) {
    console.log("\n=== EMAIL FAILED; CODE BELOW ===");
    console.log("Email:", email);
    console.log("Code :", verificationCode);
    console.log("Error:", err.message);
    console.log("================================\n");
    return { success: true };
  }
};

export const sendResendVerificationEmail = sendVerificationEmail;