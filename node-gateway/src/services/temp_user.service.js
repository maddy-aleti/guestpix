// src/services/temp_user.service.js
import TempUser from "../models/tempUser.model.js";

export const generateVerificationCode = () => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

export const getCodeExpiry = () => {
  return new Date(Date.now() + 10 * 60 * 1000); // 10 minutes
};

export const createTempUser = async (email, role) => {
  try {
    // Check if email already exists in TempUser
    const existingTempUser = await TempUser.findOne({ email: email.toLowerCase() });
    if (existingTempUser) {
      throw new Error("Email already registered");
    }

    const tempUser = await TempUser.create({
      email: email.toLowerCase(),
      role: role,
      isVerified: false
    });

    return tempUser;
  } catch (err) {
    throw err;
  }
};

export const updateVerificationCode = async (tempUserId, verificationCode) => {
  try {
    const tempUser = await TempUser.findByIdAndUpdate(
      tempUserId,
      {
        verificationCode,
        verificationCodeExpiry: getCodeExpiry()
      },
      { new: true }
    );
    return tempUser;
  } catch (err) {
    throw err;
  }
};

export const verifyTempUserEmail = async (tempUserId, verificationCode) => {
  try {
    const tempUser = await TempUser.findById(tempUserId);

    if (!tempUser) {
      throw new Error("User not found");
    }

    if (tempUser.verificationCode !== verificationCode) {
      throw new Error("Invalid verification code");
    }

    if (new Date() > tempUser.verificationCodeExpiry) {
      throw new Error("Verification code expired");
    }

    const updatedTempUser = await TempUser.findByIdAndUpdate(
      tempUserId,
      {
        isVerified: true,
        verificationCode: null,
        verificationCodeExpiry: null
      },
      { new: true }
    );

    return updatedTempUser;
  } catch (err) {
    throw err;
  }
};

export const deleteTempUser = async (tempUserId) => {
  try {
    await TempUser.findByIdAndDelete(tempUserId);
  } catch (err) {
    throw err;
  }
};