import express from "express";
import dotenv from "dotenv";
dotenv.config(); 
import faceRoutes from "./routes/face.routes.js";
import tempUserRoutes from "./routes/temp_user.routes.js";
import photographerRoutes from "./routes/photographer.routes.js";
import connectDB from "./config/db.js";

const app = express();
connectDB();
app.use(express.json());

app.use("/api/face", faceRoutes);
app.use("/api/temp-user", tempUserRoutes);
app.use("/api/photographer", photographerRoutes);

app.listen(3000, () => {
  console.log("Node API running on port 3000");
});