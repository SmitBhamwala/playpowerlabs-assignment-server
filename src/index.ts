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
    origin: ["https://playpowerlabs-assignment.vercel.app"],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"]
  })
);

// Serve static files
app.use("/uploads", express.static("uploads"));

const PORT = Number(process.env.PORT) || 5001;

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

/**
 * Extracts text per page using pdf-parse's pagerender.
 * Returns [{ pageNumber, text }, ...] in correct order.
 */
async function extractTextByPage(
  pdfPath: string
): Promise<Array<{ pageNumber: number; text: string }>> {
  const dataBuffer = fs.readFileSync(pdfPath);

  // We'll collect per-page text here in order
  const pages: string[] = [];

  await pdfParse(dataBuffer, {
    // Called once per page, sequentially
    pagerender: async (pageData: any) => {
      const textContent = await pageData.getTextContent();
      let lastY: number | undefined;
      let pageText = "";

      for (const item of textContent.items) {
        const str: string = item.str || "";
        const y: number | undefined = item.transform?.[5];

        if (lastY === undefined || lastY === y) {
          pageText += str + " ";
        } else {
          pageText += "\n" + str + " ";
        }
        lastY = y;
      }

      // Normalize whitespace
      pageText = pageText.replace(/\s+/g, " ").trim();

      // Push to our per-page array (order preserved)
      pages.push(pageText);

      // Must return the page text for pdf-parse's own aggregate behavior
      return pageText;
    }
  });

  // Map to [{pageNumber, text}] with 1-based page numbers
  return pages.map((text, idx) => ({ pageNumber: idx + 1, text }));
}

app.get("/", (req: Request, res: Response) => {
  res.send("PDF Q&A with Gemini API - Server is running");
});

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

      // const fileBuffer = fs.readFileSync(req.file.path);

      const pdfData = await extractTextByPage(req.file.path);

      if (!pdfData) {
        return res
          .status(500)
          .json({ error: "Failed to extract text from PDF" });
      }

      const pdfId = uuidv4();
      const embeddings: EmbeddingEntry[] = [];

      // Process each page separately
      for (let i = 0; i < pdfData.length; i++) {
        const pageNumber = pdfData[i].pageNumber;
        const pageText = pdfData[i].text.replace(/\s+/g, " ").trim();

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
      const fileUrl = `http://3.6.41.198/uploads/${req.file.filename}`;

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

    const queryEmbeddingResp = await embeddingModel.embedContent(question);
    const queryEmbedding = queryEmbeddingResp.embedding?.values || [];

    const ranked = embeddings
      .map((e) => ({
        ...e,
        score: cosineSimilarity(queryEmbedding, e.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    const context = ranked
      .map((r, i) => `(${i + 1}) [p.${r.pageNumber}] ${r.text}`)
      .join("\n");

    // Stream answer from Gemini
    const result = await model.generateContentStream({
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `You are given the following context from a PDF:\n\n${context}\n\n
                    Question: ${question}\n\n
                    Instructions:
                    1. Answer the question clearly.
                    2. Answer should be in a markdown format.
                    3. If needed, break down the answer in proper paragraphs with headings and subheadings.
                    4. Use can use bullet points or numbered lists if needed.
                    5. Don't make up answers. If the answer is not contained within the context, say "I don't know. Answer is not in the provided PDF".
                    6. At the very end, output the exact pagenumber citations you used for the answer in this format ONLY:
                    "Citations: [2, 3]".`
            }
          ]
        }
      ]
    });

    res.setHeader("Content-Type", "text/event-stream");

    let citations: number[] = [];

    for await (const chunk of result.stream) {
      const text = chunk.text();

      // Try to extract citations as soon as they appear
      const citationMatch = text.match(/Citations:\s*\[\s*([^\]]*)\s*\]/i);
      if (citationMatch) {
        citations = citationMatch[1]
          .split(",")
          .map((n) => parseInt(n.trim(), 10))
          .filter((n) => !isNaN(n));
      }

      //Remove citation from the text chunk
      const cleanText = text.replace(/Citations:\s*\[\s*([^\]]*)\s*\]/gi, "");

      res.write(
        `data: ${JSON.stringify({
          text: cleanText,
          citations
        })}`
      );
    }

    res.end();
  } catch (err) {
    res.status(500).json({ error: "Query failed" });
  }
});

app.listen(PORT, "0.0.0.0", () =>
  console.log(`Server running on port ${PORT}`)
);
