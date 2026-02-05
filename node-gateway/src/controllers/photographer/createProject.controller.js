// src/controllers/photographer/createProject.controller.js
import { createProject } from "../../services/photographer.service.js";

export const createProjectController = async (req, res) => {
  try {
    const { projectName, eventOwnerName, eventName, date, photographerId } = req.body;
    // TODO: After middleware is implemented, get photographerId from req.photographerId instead of req.body

    // Validate required fields
    if (!projectName || !eventOwnerName || !eventName || !date || !photographerId) {
      return res.status(400).json({
        success: false,
        message: "All fields are required (projectName, eventOwnerName, eventName, date, photographerId)"
      });
    }

    // Validate date format
    const projectDate = new Date(date);
    if (isNaN(projectDate.getTime())) {
      return res.status(400).json({
        success: false,
        message: "Invalid date format"
      });
    }

    const project = await createProject({
      projectName,
      eventOwnerName,
      eventName,
      date: projectDate,
      photographerId
    });

    return res.status(201).json({
      success: true,
      message: "Project created successfully",
      data: {
        projectId: project._id,
        projectName: project.projectName,
        eventOwnerName: project.eventOwnerName,
        eventName: project.eventName,
        date: project.date,
        status: project.status,
        createdAt: project.createdAt
      }
    });
  } catch (err) {
    console.error("Error in createProject:", err);
    return res.status(500).json({
      success: false,
      message: err.message || "Failed to create project"
    });
  }
};
