// src/models/project.model.js
import mongoose from "mongoose";

const projectSchema = new mongoose.Schema(
  {
    projectName: {
      type: String,
      required: true,
      trim: true
    },

    eventOwnerName: {
      type: String,
      required: true,
      trim: true
    },

    eventName: {
      type: String,
      required: true,
      trim: true
    },

    date: {
      type: Date,
      required: true
    },

    photographer: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Photographer",
      required: true
    },

    status: {
      type: String,
      enum: ["active", "completed", "archived"],
      default: "active"
    }
  },
  { timestamps: true }
);

export default mongoose.model("Project", projectSchema);
