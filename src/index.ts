import dotenv from "dotenv";
dotenv.config();

import cors from "cors";

import { GoogleGenerativeAI } from "@google/generative-ai";
import express, { Request, Response } from "express";
import fs from "fs";
import multer from "multer";
import pdfParse from "pdf-parse";
import { v4 as uuidv4 } from "uuid";

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(express.json());

app.use(
  cors({
    origin: ["http://localhost:5173"]
  })
);

// Serve static files
app.use("/uploads", express.static("uploads"));

const PORT = process.env.PORT || 5001;

// Type for embedding data
interface EmbeddingEntry {
  text: string;
  embedding: number[];
  pageNumber: number;
}

// In-memory vector store: { pdfId: [ { text, embedding } ] }
const vectorStores = new Map<string, EmbeddingEntry[]>();

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY1 as string);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-001" });
const genTextAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY2 as string);
const embeddingModel = genTextAI.getGenerativeModel({
  model: "text-embedding-004"
});

/**
 * Utility: Cosine similarity
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dot = 0.0,
    normA = 0.0,
    normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] ** 2;
    normB += vecB[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function chunkText(text: string, chunkSize = 500, overlap = 50): string[] {
  const sentences = text.split(/(?<=[.?!])\s+/); // split by sentence
  const chunks: string[] = [];
  let currentChunk = "";

  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > chunkSize) {
      chunks.push(currentChunk.trim());
      // carry over some overlap for context
      currentChunk = currentChunk.slice(-overlap) + " " + sentence;
    } else {
      currentChunk += " " + sentence;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

/**
 * Upload PDF → parse → chunk → embed → save
 */
app.post(
  "/upload",
  upload.single("pdf"),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      const fileBuffer = fs.readFileSync(req.file.path);
      const pdfData = await pdfParse(fileBuffer);

      // Split by page using form feed ("\f")
      const pages = pdfData.text.split("\f");

      const pdfId = uuidv4();
      const embeddings: EmbeddingEntry[] = [];

      // Process each page separately
      for (let i = 0; i < pages.length; i++) {
        const pageNumber = i + 1;
        const pageText = pages[i].replace(/\s+/g, " ").trim();

        if (!pageText) continue;

        const response = await embeddingModel.embedContent(pageText);

        embeddings.push({
          text: pageText,
          embedding: response.embedding?.values || [],
          pageNumber
        });
      }

      // Store in memory vector store with pdfId
      vectorStores.set(pdfId, embeddings);

      // File URL for client preview/download
      const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${
        req.file.filename
      }`;

      res.json({ pdfId, fileUrl });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: "PDF processing failed" });
    }
  }
);

/**
 * Ask question → embed query → similarity search → stream Gemini response
 */
app.post("/ask", async (req: Request, res: Response) => {
  try {
    const { pdfId, question } = req.body as { pdfId: string; question: string };
    const embeddings = vectorStores.get(pdfId);

    if (!embeddings) {
      return res.status(404).json({ error: "PDF not found" });
    }

    // Embed user query
    const queryEmbeddingResp = await embeddingModel.embedContent(question);
    const queryEmbedding = queryEmbeddingResp.embedding?.values || [];

    // Rank chunks
    const ranked = embeddings
      .map((e) => ({
        ...e,
        score: cosineSimilarity(queryEmbedding, e.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    const uniqueCitations = [...new Set(ranked.map((r) => r.pageNumber))];

    const context = ranked
      .map((r, i) => `(${i + 1}) [p.${r.pageNumber}] ${r.text}`)
      .join("\n");
    console.log("Context for Gemini:", context);

    // Stream answer from Gemini
    const result = await model.generateContentStream({
      contents: [
        {
          role: "user",
          parts: [{ text: `Context: ${context}\n\nQuestion: ${question}` }]
        }
      ]
    });

    res.setHeader("Content-Type", "text/event-stream");

    for await (const chunk of result.stream) {
      res.write(
        `data: ${JSON.stringify({
          text: chunk.text(),
          citations: uniqueCitations
        })}\n\n`
      );
    }

    res.end();
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Query failed" });
  }
});

app.listen(PORT, () =>
  console.log(`🚀 Server running on http://localhost:${PORT}`)
);
