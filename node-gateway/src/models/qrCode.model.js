// src/models/qrCode.model.js
import mongoose from "mongoose";

const qrCodeSchema = new mongoose.Schema(
  {
    project: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Project",
      required: true
    },

    photographer: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Photographer",
      required: true
    },

    qrType: {
      type: String,
      enum: ["user", "client"],
      required: true
    },

    qrCode: {
      type: String,
      required: true,
      unique: true
    },

    qrImageUrl: {
      type: String,
      required: true
    },

    expiresAt: {
      type: Date,
      required: true
    },

    isActive: {
      type: Boolean,
      default: true
    },

    accessCount: {
      type: Number,
      default: 0
    }
  },
  { timestamps: true }
);

// Index to automatically remove expired QR codes
qrCodeSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

export default mongoose.model("QRCode", qrCodeSchema);