import dotenv from "dotenv";
dotenv.config();

import cors from "cors";
import express, { Request, Response } from "express";
import multer from "multer";
import path from "path";
import { v4 as uuidv4 } from "uuid";

const app = express();
const PORT = process.env.PORT || 5000;

// allow frontend
app.use(
  cors({
    origin: ["http://localhost:5173"]
  })
);
app.use(express.json());

// configure multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, uuidv4() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// serve static files
app.use("/uploads", express.static("uploads"));

// store PDFs in memory (for demo)
const pdfStore: Record<string, { filename: string; url: string }> = {};

// Create a type that extends Request to include Multer's file
interface MulterRequest extends Request {
  file: Express.Multer.File;
}

app.post("/upload", upload.single("pdf"), (req: Request, res: Response) => {
  const file = (req as MulterRequest).file;

  if (!file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  // Generate unique ID for this PDF
  const pdfId = uuidv4();

  const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${
    file.filename
  }`;
  console.log("FileURL: ", fileUrl);

  // store reference
  pdfStore[pdfId] = {
    filename: file.originalname,
    url: fileUrl
  };

  res.json({
    message: "PDF uploaded successfully",
    pdfId,
    filename: file.originalname,
    fileUrl
  });
});

app.post("/ask", express.json(), (req: Request, res: Response) => {
  const { pdfId, question } = req.body;

  if (!pdfId || !question) {
    return res.status(400).json({ error: "pdfId and question are required" });
  }

  const pdf = pdfStore[pdfId];
  if (!pdf) {
    return res.status(404).json({ error: "PDF not found" });
  }

  // Here you’ll load/process the PDF file from pdf.url or uploads folder
  // then run your Q&A logic on it.
  // For now, just return confirmation
  res.json({ answer: `Pretend answer for "${question}"` });
});

app.listen(PORT, () =>
  console.log("Server is running on http://localhost:5000")
);
