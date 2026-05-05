import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";
import { Document } from "langchain/document";

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

class MemoryManager {
  private static instance: MemoryManager;
  private history: Map<string, Array<{ score: number; member: string }>>;
  private vectorDBClient: PineconeClient | SupabaseClient;

  public constructor() {
    this.history = new Map();
    if (process.env.VECTOR_DB === "pinecone") {
      this.vectorDBClient = new PineconeClient();
    } else {
      const auth = {
        detectSessionInUrl: false,
        persistSession: false,
        autoRefreshToken: false,
      };
      const url = process.env.SUPABASE_URL!;
      const privateKey = process.env.SUPABASE_PRIVATE_KEY!;
      this.vectorDBClient = createClient(url, privateKey, { auth });
    }
  }

  public async init() {
    if (this.vectorDBClient instanceof PineconeClient) {
      await this.vectorDBClient.init({
        apiKey: process.env.PINECONE_API_KEY!,
        environment: process.env.PINECONE_ENVIRONMENT!,
      });
    }
  }

  private sanitizeString(input: string): string {
    return input
      .trim()
      .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
      .replace(/[<>;"'`\\]/g, "");
  }

  private sanitizeInput(input: string, maxLength: number = 4096): string {
    let sanitized = input.replace(/\0/g, "");
    sanitized = sanitized.trim();
    if (sanitized.length > maxLength) {
      sanitized = sanitized.substring(0, maxLength);
    }
    return sanitized;
  }

  private sanitizeOutputDocs(docs: Document[] | void): Document[] | void {
    if (!docs) return docs;
    const dangerousPatterns = [
      /\beval\s*\(/gi,
      /new\s+Function\s*\(/gi,
      /import\s*\(/gi,
      /require\s*\(/gi,
      /setTimeout\s*\(\s*["'`]/gi,
      /setInterval\s*\(\s*["'`]/gi,
      /execScript\s*\(/gi,
      /\bFunction\s*\(/gi,
    ];
    return docs.map((doc) => {
      let content = doc.pageContent;
      for (const pattern of dangerousPatterns) {
        content = content.replace(pattern, "[REMOVED]");
      }
      return new Document({ pageContent: content, metadata: doc.metadata });
    });
  }

  public async vectorSearch(
    recentChatHistory: string,
    companionFileName: string
  ) {
    const sanitizedHistory = this.sanitizeInput(
      this.sanitizeString(recentChatHistory)
    );
    const sanitizedFileName = this.sanitizeInput(
      this.sanitizeString(companionFileName)
    );

    if (process.env.VECTOR_DB === "pinecone") {
      console.log("INFO: using Pinecone for vector search.");
      const pineconeClient = <PineconeClient>this.vectorDBClient;

      const pineconeIndex = pineconeClient.Index(
        process.env.PINECONE_INDEX! || ""
      );

      const vectorStore = await PineconeStore.fromExistingIndex(
        new HuggingFaceInferenceEmbeddings({
          apiKey: process.env.HUGGINGFACEHUB_API_KEY,
        }),
        { pineconeIndex }
      );

      const similarDocs = await vectorStore
        .similaritySearch(sanitizedHistory, 3, {
          fileName: sanitizedFileName,
        })
        .catch((err) => {
          console.log("WARNING: failed to get vector search results.", err);
        });
      return this.sanitizeOutputDocs(similarDocs);
    } else {
      console.log("INFO: using Supabase for vector search.");
      const supabaseClient = <SupabaseClient>this.vectorDBClient;
      const vectorStore = await SupabaseVectorStore.fromExistingIndex(
        new HuggingFaceInferenceEmbeddings({
          apiKey: process.env.HUGGINGFACEHUB_API_KEY,
        }),
        {
          client: supabaseClient,
          tableName: "documents",
          queryName: "match_documents",
        }
      );
      const similarDocs = await vectorStore
        .similaritySearch(sanitizedHistory, 3)
        .catch((err) => {
          console.log("WARNING: failed to get vector search results.", err);
        });
      return this.sanitizeOutputDocs(similarDocs);
    }
  }

  public static async getInstance(): Promise<MemoryManager> {
    if (!MemoryManager.instance) {
      MemoryManager.instance = new MemoryManager();
      await MemoryManager.instance.init();
    }
    return MemoryManager.instance;
  }

  private generateRedisCompanionKey(companionKey: CompanionKey): string {
    return `${companionKey.companionName}-${companionKey.modelName}-${companionKey.userId}`;
  }

  public async writeToHistory(text: string, companionKey: CompanionKey) {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const sanitizedText = this.sanitizeString(this.sanitizeInput(text));
    const key = this.generateRedisCompanionKey(companionKey);

    if (!this.history.has(key)) {
      this.history.set(key, []);
    }
    const entries = this.history.get(key)!;
    entries.push({ score: Date.now(), member: sanitizedText });
    this.history.set(key, entries);

    return entries.length;
  }

  public async readLatestHistory(companionKey: CompanionKey): Promise<string> {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    const entries = this.history.get(key) || [];

    const sorted = [...entries].sort((a, b) => a.score - b.score);
    const result = sorted.map((e) => e.member);
    const sliced = result.slice(-30).reverse();
    const recentChats = sliced.reverse().join("\n");
    return recentChats;
  }

  public async seedChatHistory(
    seedContent: String,
    delimiter: string = "\n",
    companionKey: CompanionKey
  ) {
    const key = this.generateRedisCompanionKey(companionKey);
    if (this.history.has(key) && this.history.get(key)!.length > 0) {
      console.log("User already has chat history");
      return;
    }

    const sanitizedSeed = this.sanitizeString(
      this.sanitizeInput(seedContent as string)
    );
    const content = sanitizedSeed.split(delimiter);
    let counter = 0;
    const entries: Array<{ score: number; member: string }> = [];
    for (const line of content) {
      entries.push({ score: counter, member: line });
      counter += 1;
    }
    this.history.set(key, entries);
  }
}

export default MemoryManager;