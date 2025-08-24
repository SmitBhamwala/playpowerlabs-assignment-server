import dotenv from "dotenv";
dotenv.config();

import cors from "cors";
import express, { Request, Response } from "express";
import multer from "multer";
import path from "path";
import { v4 as uuidv4 } from "uuid";

// LlamaIndex + LlamaParse
import { GoogleGenAI } from "@google/genai";
import {
  BaseEmbedding,
  BaseEmbeddingOptions,
  ChatResponseChunk,
  LlamaParseReader,
  MessageContentDetail,
  Settings,
  VectorStoreIndex,
  
} from "llamaindex";

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY1 });
const textEmbeddingAI = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY2
});

class GeminiEmbedding extends BaseEmbedding {
  embedBatchSize = 1; // required property
  id = "gemini:text-embedding-004"; // required property

  constructor() {
    // call BaseEmbedding constructor
    super();
  }

  async getTextEmbedding(text: string): Promise<number[]> {
    const result = await textEmbeddingAI.models.embedContent({
      model: "text-embedding-004",
      contents: [{ role: "user", parts: [{ text }] }]
    });

    if (!result.embeddings || result.embeddings.length === 0) {
      throw new Error("No embeddings returned from Gemini API");
    }

    // take the first embedding (since we only passed one input)
    return result.embeddings[0].values!;
  }

  async getNodeEmbedding(nodes: any[]): Promise<any[]> {
    const embeddings = await this.getTextEmbeddings(
      nodes.map((n) => n.getContent()) // node text
    );
    return nodes.map((n, i) => {
      n.embedding = embeddings[i];
      return n;
    });
  }

  // required abstract: embed node objects and return them
  // async getNodeEmbedding<Options extends Record<string, unknown>>(
  //   nodes: BaseNode<Metadata>[],
  //   _options?: Options
  // ): Promise<BaseNode<Metadata>[]> {
  //   const contents = nodes.map((n) => n.getContent());
  //   const vectors = await this.getTextEmbeddings(contents);
  //   return nodes.map((n, i) => {
  //     (n as any).embedding = vectors[i]; // assign vector
  //     return n;
  //   });
  // }

  getTextEmbeddingsBatch(
    texts: string[],
    _options?: BaseEmbeddingOptions
  ): Promise<Array<number[]>> {
    return this.getTextEmbeddings(texts);
  }

  async getQueryEmbedding(
    query: MessageContentDetail
  ): Promise<number[] | null> {
    if ("text" in query) {
      return this.getTextEmbedding(query.text);
    }
    return null;
  }

  // (Optional, but avoids TS complaints)
  // async getTextEmbeddings(texts: string[]): Promise<number[][]> {
  //   return Promise.all(texts.map((t) => this.getTextEmbedding(t)));
  // }

  similarity(a: number[], b: number[]): number {
    // cosine similarity
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (magA * magB);
  }

  truncateMaxTokens(input: string[]): string[] {
    return input;
  }
}

// tell LlamaIndex to use Gemini embeddings
Settings.embedModel = new GeminiEmbedding();

// Wrap Gemini as an LLM for LlamaIndex
Settings.llm = {
  chat: async (params) => {
    if ("messages" in params) {
      // non-streaming case
      const messages = params.messages;
      const prompt = messages.map((m) => `${m.role}: ${m.content}`).join("\n");

      const result = await genAI.models.generateContent({
        model: "gemini-2.0-flash-001",
        contents: prompt
      });

      // const response: ChatResponse = {
      //   message: {
      //     role: "assistant",
      //     content: result.text
      //   }
      // };
      return result.text;
    }

    // streaming case
    const stream = await genAI.models.generateContentStream({
      model: "gemini-2.0-flash-001",
      contents: {
        role: "user",
        parts: [{ text: prompt }]
      }
    });

    async function* streamIterator(): AsyncIterable<ChatResponseChunk> {
      for await (const chunk of stream) {
        yield {
          delta: chunk.text,
          raw: 
        };
      }
    }

    return streamIterator();
  }
  // complete: async (params) => {
  //   const prompt = params.prompt;

  //   const result = await genAI.models.generateContent({
  //     model: "gemini-2.0-flash-001",
  //     contents: [{ role: "user", parts: [{ text: prompt }] }]
  //   });

  //   return {
  //     message: {
  //       role: "assistant",
  //       content: result.text
  //     }
  //   };
  // }
};

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
const pdfStore: Record<
  string,
  { fileName: string; url: string; index?: VectorStoreIndex }
> = {};

// Create a type that extends Request to include Multer's file
interface MulterRequest extends Request {
  file: Express.Multer.File;
}

// Upload + parse PDF
app.post(
  "/upload",
  upload.single("pdf"),
  async (req: Request, res: Response) => {
    const file = (req as MulterRequest).file;

    if (!file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // Generate unique ID for this PDF
    const pdfId = uuidv4();

    const filePath = path.resolve("uploads", file.filename);

    const fileUrl = `${req.protocol}://${req.get("host")}/uploads/${
      file.filename
    }`;

    try {
      // Parse PDF with LlamaParse
      const reader = new LlamaParseReader({
        apiKey: process.env.LLAMA_CLOUD_API_KEY,
        resultType: "markdown" // "json" or "text" also possible
      });

      const docs = await reader.loadData(filePath);

      // Build vector index
      const index = await VectorStoreIndex.fromDocuments(docs);

      // store reference
      pdfStore[pdfId] = {
        fileName: file.originalname,
        url: fileUrl,
        index
      };

      res.json({
        message: "PDF uploaded and indexed successfully",
        pdfId,
        fileName: file.originalname,
        fileUrl
      });
    } catch (error) {
      console.error("Error parsing PDF:", error);
      res.status(500).json({ error: "Failed to parse and index PDF" });
    }
  }
);

app.post("/ask", express.json(), async (req: Request, res: Response) => {
  const { pdfId, question } = req.body;

  if (!pdfId || !question) {
    return res.status(400).json({ error: "pdfId and question are required" });
  }

  const pdf = pdfStore[pdfId];
  if (!pdf || !pdf.index) {
    return res.status(404).json({ error: "PDF not found or not indexed" });
  }

  try {
    const queryEngine = pdf.index.asQueryEngine();
    const response = await queryEngine.query(question);

    let finalAnswer = "";
    for await (const chunk of response) {
      finalAnswer += chunk.message ?? "";
    }

    res.json({ answer: finalAnswer });
  } catch (error) {
    console.error("Error querying PDF:", error);
    res.status(500).json({ error: "Failed to query PDF" });
  }
});

app.listen(PORT, () =>
  console.log("Server is running on http://localhost:5000")
);
