// Call embeding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { CharacterTextSplitter } from "langchain/text_splitter";

import fs from "fs";
import path from "path";

dotenv.config({ path: `.env.local` });

// Validate required environment variables
const requiredEnvVars = ["SUPABASE_URL", "SUPABASE_PRIVATE_KEY", "HUGGINGFACEHUB_API_KEY"];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
}

const MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024; // 1 MB
const MAX_CONTENT_LENGTH = 500000; // max characters for content
const COMPANIONS_DIR = path.resolve("companions");

// PII redaction patterns (general)
function redactPII(text) {
  // Email addresses
  text = text.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");
  // Phone numbers (general)
  text = text.replace(/(\+?\d[\d\s\-().]{7,}\d)/g, "[REDACTED_PHONE]");
  // SSNs (US)
  text = text.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Credit card numbers
  text = text.replace(/\b(?:\d[ -]?){13,16}\b/g, "[REDACTED_CC]");
  // IP addresses
  text = text.replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[REDACTED_IP]");
  return text;
}

// Singapore PII detection and redaction
function redactSingaporePII(text) {
  // NRIC/FIN numbers (e.g., S1234567A, T1234567B, F1234567C, G1234567D)
  if (/\b[STFG]\d{7}[A-Z]\b/i.test(text)) {
    console.warn("Warning: Singapore NRIC/FIN detected and redacted.");
    text = text.replace(/\b[STFG]\d{7}[A-Z]\b/gi, "[REDACTED_NRIC]");
  }
  // SingPass identifiers (heuristic: "singpass" followed by identifier)
  if (/singpass\s*[:\-]?\s*\S+/i.test(text)) {
    console.warn("Warning: SingPass identifier detected and redacted.");
    text = text.replace(/singpass\s*[:\-]?\s*\S+/gi, "[REDACTED_SINGPASS]");
  }
  // Singapore phone numbers (+65 XXXX XXXX or 8/9 XXXX XXXX)
  if (/(\+65[\s\-]?)?[89]\d{3}[\s\-]?\d{4}\b/.test(text)) {
    console.warn("Warning: Singapore phone number detected and redacted.");
    text = text.replace(/(\+65[\s\-]?)?[89]\d{3}[\s\-]?\d{4}\b/g, "[REDACTED_SG_PHONE]");
  }
  // Singapore postal codes (6-digit starting with valid range)
  if (/\b[0-9]{6}\b/.test(text)) {
    console.warn("Warning: Potential Singapore postal code detected and redacted.");
    text = text.replace(/\b[0-9]{6}\b/g, "[REDACTED_POSTAL]");
  }
  return text;
}

// Sanitize and validate input to AI model
function sanitizeInput(text) {
  // Strip null bytes and non-printable control characters (except newline, tab, carriage return)
  text = text.replace(/\0/g, "");
  text = text.replace(/[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]/g, "");

  // Enforce maximum content length
  if (text.length > MAX_CONTENT_LENGTH) {
    text = text.substring(0, MAX_CONTENT_LENGTH);
  }

  // Detect and reject potential prompt injection patterns
  const promptInjectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+/i,
    /act\s+as\s+(if\s+you\s+are|a)\s+/i,
    /system\s*:\s*/i,
    /\[INST\]/i,
    /<\|im_start\|>/i,
  ];
  for (const pattern of promptInjectionPatterns) {
    if (pattern.test(text)) {
      throw new Error("Potential prompt injection detected in content. Rejecting document.");
    }
  }

  return text;
}

// Sanitize file content for malicious patterns (prompt injection, hidden chars, base64, shell commands)
function sanitizeFileContent(text) {
  // Strip hidden/invisible characters (zero-width, soft hyphen, etc.)
  text = text.replace(/[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]/g, "");

  // Detect base64-encoded content (long base64 strings)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/g;
  if (base64Pattern.test(text)) {
    console.warn("Warning: Potential base64-encoded content detected and stripped.");
    text = text.replace(/(?:[A-Za-z0-9+/]{40,}={0,2})/g, "[REDACTED_BASE64]");
  }

  // Detect leetspeak patterns (simple heuristic)
  const leetspeakPattern = /(\b\w*[013457@$!]\w*\b){5,}/;
  if (leetspeakPattern.test(text)) {
    console.warn("Warning: Potential leetspeak content detected.");
  }

  // Detect shell/binary commands
  const shellPatterns = [
    /\b(bash|sh|zsh|cmd|powershell|exec|system|popen|subprocess|os\.system)\s*[\(\[]/i,
    /\b(rm\s+-rf|chmod|chown|wget|curl\s+.*\|)\b/i,
    /\$\(.*\)/,
    /`[^`]+`/,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(text)) {
      console.warn("Warning: Potential shell/binary command detected and stripped.");
      text = text.replace(pattern, "[REDACTED_CMD]");
    }
  }

  // Detect suspicious instruction patterns
  const suspiciousPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+/i,
    /act\s+as\s+(if\s+you\s+are|a)\s+/i,
  ];
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(text)) {
      console.warn("Warning: Suspicious instruction pattern detected and stripped.");
      text = text.replace(pattern, "[REDACTED_INJECTION]");
    }
  }

  return text;
}

// Validate and sanitize LLM output (check for dynamic code execution primitives)
function sanitizeLLMOutput(text) {
  const dangerousPatterns = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bsubprocess\b/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*["'`]/i,
    /\bsetInterval\s*\(\s*["'`]/i,
    /\bimportlib\b/i,
    /\b__import__\s*\(/i,
    /\bos\.system\s*\(/i,
    /\bchild_process\b/i,
  ];
  for (const pattern of dangerousPatterns) {
    if (pattern.test(text)) {
      throw new Error(`Dangerous code execution primitive detected in LLM output: ${pattern}`);
    }
  }
  return text;
}

// Path traversal protection
function safeResolvePath(dir, fileName) {
  const resolvedDir = path.resolve(dir);
  const resolvedFile = path.resolve(dir, fileName);
  if (!resolvedFile.startsWith(resolvedDir + path.sep) && resolvedFile !== resolvedDir) {
    throw new Error(`Path traversal detected for file: ${fileName}`);
  }
  return resolvedFile;
}

const fileNames = fs.readdirSync("companions");
const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

const langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
    if (fileName.endsWith(".txt")) {
      const filePath = safeResolvePath(COMPANIONS_DIR, fileName);

      // File size check
      const stats = fs.statSync(filePath);
      if (stats.size > MAX_FILE_SIZE_BYTES) {
        console.warn(`Skipping file ${fileName}: exceeds maximum allowed size of ${MAX_FILE_SIZE_BYTES} bytes.`);
        return undefined;
      }

      let fileContent = fs.readFileSync(filePath, "utf8");

      // Truncate to safe maximum length
      if (fileContent.length > MAX_CONTENT_LENGTH) {
        fileContent = fileContent.substring(0, MAX_CONTENT_LENGTH);
      }

      // Sanitize file content for malicious patterns
      fileContent = sanitizeFileContent(fileContent);

      // Redact general PII
      fileContent = redactPII(fileContent);

      // Redact Singapore-specific PII
      fileContent = redactSingaporePII(fileContent);

      const lastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];

      // Sanitize and validate input before passing to AI model
      let sanitizedSection;
      try {
        sanitizedSection = sanitizeInput(lastSection);
      } catch (err) {
        console.warn(`Skipping file ${fileName}: ${err.message}`);
        return undefined;
      }

      const splitDocs = await splitter.createDocuments([sanitizedSection]);
      return splitDocs.map((doc) => {
        // Validate and sanitize LLM output
        let safeContent;
        try {
          safeContent = sanitizeLLMOutput(doc.pageContent);
        } catch (err) {
          console.warn(`Skipping document chunk from ${fileName}: ${err.message}`);
          return undefined;
        }
        return new Document({
          metadata: { fileName },
          pageContent: safeContent,
        });
      });
    }
  })
);

const auth = {
  detectSessionInUrl: false,
  persistSession: false,
  autoRefreshToken: false,
};

const client = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_PRIVATE_KEY,
  { auth }
);

const flatDocs = langchainDocs.flat().filter((doc) => doc !== undefined);

await SupabaseVectorStore.fromDocuments(
  flatDocs,
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  }),
  {
    client,
    tableName: "documents",
  }
);