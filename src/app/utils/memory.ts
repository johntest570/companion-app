import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";
import { createHmac, randomInt } from "crypto";
import { Document } from "langchain/document";

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

export interface HistoryStore {
  zadd(key: string, entry: { score: number; member: string }): Promise<any>;
  zrange(key: string, min: number, max: number, opts?: { byScore?: boolean }): Promise<string[]>;
  exists(key: string): Promise<number | boolean>;
  expire(key: string, seconds: number): Promise<any>;
}

const SESSION_TTL_SECONDS = 60 * 60 * 24; // 24 hours
const MAX_INPUT_LENGTH = 4096;

function getHmacSecret(): string {
  const secret = process.env.REDIS_KEY_SECRET;
  if (!secret) {
    throw new Error("REDIS_KEY_SECRET environment variable is not set");
  }
  return secret;
}

function signKey(raw: string): string {
  const secret = getHmacSecret();
  const mac = createHmac("sha256", secret).update(raw).digest("hex");
  return mac;
}

function encodeKeyComponent(component: string): string {
  return encodeURIComponent(String(component));
}

function sanitizeInput(input: string): string {
  if (typeof input !== "string") {
    throw new Error("Input must be a string");
  }

  // Strip null bytes
  let sanitized = input.replace(/\0/g, "");

  // Trim whitespace
  sanitized = sanitized.trim();

  // Enforce maximum length
  if (sanitized.length > MAX_INPUT_LENGTH) {
    sanitized = sanitized.slice(0, MAX_INPUT_LENGTH);
  }

  // Reject empty strings
  if (sanitized.length === 0) {
    throw new Error("Input must not be empty");
  }

  // Reject shell command patterns
  const shellPatterns = [
    /\$\(.*\)/,
    /`[^`]*`/,
    /;\s*(rm|ls|cat|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
    /\|\s*(rm|ls|cat|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
    /&&\s*(rm|ls|cat|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("Input contains potentially malicious shell command patterns");
    }
  }

  // Reject base64-encoded payloads (heuristic: long base64 strings)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/;
  if (base64Pattern.test(sanitized)) {
    // Attempt to decode and check for binary/executable content
    const b64matches = sanitized.match(/[A-Za-z0-9+/]{40,}={0,2}/g) || [];
    for (const match of b64matches) {
      try {
        const decoded = Buffer.from(match, "base64").toString("utf8");
        if (/eval|exec|Function|system|spawn|shell/i.test(decoded)) {
          throw new Error("Input contains base64-encoded malicious content");
        }
      } catch (e: any) {
        if (e.message && e.message.includes("base64-encoded malicious content")) {
          throw e;
        }
        // Not valid base64, ignore
      }
    }
  }

  // Reject hidden prompt injection markers
  const injectionPatterns = [
    /\[INST\]/i,
    /<<SYS>>/i,
    /<\|system\|>/i,
    /###\s*instruction/i,
    /ignore\s+previous\s+instructions/i,
    /you\s+are\s+now/i,
  ];
  for (const pattern of injectionPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("Input contains prompt injection markers");
    }
  }

  // Reject binary/executable content
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(sanitized)) {
    throw new Error("Input contains binary or control characters");
  }

  return sanitized;
}

const DYNAMIC_CODE_PATTERNS = [
  /\beval\s*\(/i,
  /\bexec\s*\(/i,
  /new\s+Function\s*\(/i,
  /setTimeout\s*\(\s*["'`]/i,
  /setInterval\s*\(\s*["'`]/i,
  /\bimportScripts\s*\(/i,
  /document\.write\s*\(/i,
  /innerHTML\s*=/i,
];

function sanitizeDocs(docs: Document[] | undefined): Document[] {
  if (!docs) return [];
  return docs.filter((doc) => {
    const content = doc.pageContent || "";
    for (const pattern of DYNAMIC_CODE_PATTERNS) {
      if (pattern.test(content)) {
        console.log("WARNING: document redacted due to dynamic code execution pattern.");
        return false;
      }
    }
    return true;
  });
}

class MemoryManager {
  private static instance: MemoryManager;
  private static initPromise: Promise<MemoryManager> | null = null;
  private history: HistoryStore;
  private vectorDBClient: PineconeClient | SupabaseClient;

  public constructor(historyStore: HistoryStore) {
    this.history = historyStore;
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

  public async vectorSearch(
    recentChatHistory: string,
    companionFileName: string
  ) {
    let sanitizedHistory: string;
    try {
      sanitizedHistory = sanitizeInput(recentChatHistory);
    } catch (err: any) {
      console.log("WARNING: recentChatHistory failed sanitization:", err.message);
      return [];
    }

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

      const rawDocs = await vectorStore
        .similaritySearch(sanitizedHistory, 3, { fileName: companionFileName })
        .catch((err: any) => {
          console.log("WARNING: failed to get vector search results.", err.message);
        });
      return sanitizeDocs(rawDocs as Document[] | undefined);
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
      const rawDocs = await vectorStore
        .similaritySearch(sanitizedHistory, 3)
        .catch((err: any) => {
          console.log("WARNING: failed to get vector search results.", err.message);
        });
      return sanitizeDocs(rawDocs as Document[] | undefined);
    }
  }

  public static async getInstance(): Promise<MemoryManager> {
    if (MemoryManager.instance) {
      return MemoryManager.instance;
    }
    if (!MemoryManager.initPromise) {
      MemoryManager.initPromise = (async () => {
        const { Redis } = await import("@upstash/redis");
        const redisClient = Redis.fromEnv();
        const instance = new MemoryManager(redisClient as unknown as HistoryStore);
        await instance.init();
        MemoryManager.instance = instance;
        return instance;
      })();
    }
    return MemoryManager.initPromise;
  }

  private generateRedisCompanionKey(companionKey: CompanionKey): string {
    const encodedName = encodeKeyComponent(companionKey.companionName);
    const encodedModel = encodeKeyComponent(companionKey.modelName);
    const encodedUser = encodeKeyComponent(companionKey.userId);
    const raw = `${encodedName}-${encodedModel}-${encodedUser}`;
    const mac = signKey(raw);
    return `${raw}-${mac}`;
  }

  public async writeToHistory(text: string, companionKey: CompanionKey) {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    let sanitizedText: string;
    try {
      sanitizedText = sanitizeInput(text);
    } catch (err: any) {
      console.log("WARNING: text failed sanitization:", err.message);
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    const result = await this.history.zadd(key, {
      score: Date.now(),
      member: sanitizedText,
    });
    await this.history.expire(key, SESSION_TTL_SECONDS);

    return result;
  }

  public async readLatestHistory(companionKey: CompanionKey): Promise<string> {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    let result = await this.history.zrange(key, 0, Date.now(), {
      byScore: true,
    });

    result = result.slice(-30).reverse();
    const recentChats = result.reverse().join("\n");

    let sanitizedChats: string;
    try {
      sanitizedChats = sanitizeInput(recentChats);
    } catch (err: any) {
      console.log("WARNING: retrieved chat history failed sanitization:", err.message);
      return "";
    }

    return sanitizedChats;
  }

  public async seedChatHistory(
    seedContent: String,
    delimiter: string = "\n",
    companionKey: CompanionKey
  ) {
    let sanitizedSeed: string;
    try {
      sanitizedSeed = sanitizeInput(seedContent as string);
    } catch (err: any) {
      console.log("WARNING: seedContent failed sanitization:", err.message);
      return;
    }

    const key = this.generateRedisCompanionKey(companionKey);
    if (await this.history.exists(key)) {
      console.log("User already has chat history");
      return;
    }

    const content = sanitizedSeed.split(delimiter);
    for (const line of content) {
      const score = randomInt(0, 2147483647);
      await this.history.zadd(key, { score, member: line });
    }
    await this.history.expire(key, SESSION_TTL_SECONDS);
  }
}

export default MemoryManager;