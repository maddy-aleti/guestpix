import express from "express";
import { selectRole } from "../controllers/temp_user/selectRole.controller.js";
import { signIn } from "../controllers/temp_user/signIn.controller.js";
import { verifyEmail } from "../controllers/temp_user/verifyEmail.controller.js";
import { resendVerificationCode } from "../controllers/temp_user/sendVerification.controller.js";
import { uploadSelfie, upload } from "../controllers/temp_user/uploadSelfie.controller.js";

const router = express.Router();

// Step 1: Select role (user only for this flow)
router.post("/select-role", selectRole);

// Step 2: Enter email → send verification code
router.post("/sign-in", signIn);

// Step 2b: Resend verification code
router.post("/resend-verification", resendVerificationCode);

// Step 3: Verify email with code
router.post("/verify-email", verifyEmail);

// Step 4: Upload selfie after verification
router.post("/upload-selfie", upload.single("selfie"), uploadSelfie);

export default router;