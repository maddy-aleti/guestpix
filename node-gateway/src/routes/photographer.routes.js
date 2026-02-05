import express from "express";
import { signup } from "../controllers/photographer/signup.controller.js";
import { signin } from "../controllers/photographer/signIn.controller.js";
import { forgotPassword } from "../controllers/photographer/forgotPassword.controller.js";
import { verifyEmailForgotPassword } from "../controllers/photographer/verifyEmailForgotPassword.controller.js";
import { resetPassword } from "../controllers/photographer/resetPassword.controller.js";
import { createProjectController } from "../controllers/photographer/createProject.controller.js";
import { generateUserQRController } from "../controllers/photographer/generateUserQR.controller.js";
import { generateClientQRController } from "../controllers/photographer/generateClientQR.controller.js";

const router = express.Router();

// Step 1: Sign Up - Create account
router.post("/signup", signup);

// Step 2: Sign In - Login
router.post("/signin", signin);

// Step 3: Forgot Password - Send verification code
router.post("/forgot-password", forgotPassword);

// Step 4: Verify Email - Verify code
router.post("/verify-email-forgot-password", verifyEmailForgotPassword);

// Step 5: Reset Password - Set new password
router.post("/reset-password", resetPassword);

// Step 6: Create Project - Create a new project after login
router.post("/create-project", createProjectController);

router.post("/generate-user-qr", generateUserQRController);

router.post("/generate-client-qr", generateClientQRController);

export default router;
