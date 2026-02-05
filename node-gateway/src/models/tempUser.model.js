// src/models/tempUser.model.js
import mongoose from "mongoose";

const tempUserSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true
    },

    role: {
      type: String,
      enum: ["user", "photographer"],
      required: true
    },

    isVerified: {
      type: Boolean,
      default: false
    },

    verificationCode: {
      type: String
    },

    verificationCodeExpiry: {
      type: Date
    },

    photoUrl: {
      type: String
    }
  },
  { timestamps: true }
);

export default mongoose.model("TempUser", tempUserSchema);