// src/models/photographer.model.js
import mongoose from "mongoose";

const photographerSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true
    },

    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true
    },

    password: {
      type: String,
      required: true
    },

    role: {
      type: String,
      enum: ["photographer"],
      default: "photographer"
    },

    verificationCode: {
      type: String
    },

    verificationCodeExpiry: {
      type: Date
    }
  },
  { timestamps: true }
);

export default mongoose.model("Photographer", photographerSchema);
