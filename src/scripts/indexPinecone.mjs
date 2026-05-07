// Major ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pinecone
import { PineconeClient } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { Document } from "langchain/document";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";

dotenv.config({ path: `.env.local` });

// Runtime check for required environment variables
const requiredEnvVars = [
  "PINECONE_API_KEY",
  "PINECONE_ENVIRONMENT",
  "PINECONE_INDEX",
  "HUGGINGFACEHUB_API_KEY",
];
for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
}

const MAX_FILE_SIZE = 1 * 1024 * 1024; // 1 MB
const MAX_CHUNK_LENGTH = 2000;

// Sanitize and validate content
function sanitizeContent(content) {
  // Strip null bytes and non-printable control characters (except newline, tab, carriage return)
  let sanitized = content.replace(/\0/g, "").replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  return sanitized;
}

function validateContent(content) {
  if (!content || content.trim().length === 0) {
    throw new Error("Content is empty after sanitization.");
  }
}

function truncateChunk(content) {
  if (content.length > MAX_CHUNK_LENGTH) {
    return content.slice(0, MAX_CHUNK_LENGTH);
  }
  return content;
}

// Redact PII
function redactPII(content) {
  // SSN
  let redacted = content.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Email addresses
  redacted = redacted.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");
  // Phone numbers (various formats)
  redacted = redacted.replace(/(\+?\d[\d\s\-().]{7,}\d)/g, "[REDACTED_PHONE]");
  // Credit card numbers (basic pattern)
  redacted = redacted.replace(/\b(?:\d[ -]?){13,16}\b/g, "[REDACTED_CC]");
  // Passport numbers (generic: letter(s) followed by digits)
  redacted = redacted.replace(/\b[A-Z]{1,2}\d{6,9}\b/g, "[REDACTED_PASSPORT]");
  // Home addresses (basic US pattern)
  redacted = redacted.replace(/\d{1,5}\s[\w\s]{1,50}(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b/gi, "[REDACTED_ADDRESS]");
  return redacted;
}

// Singapore-specific PII detection
function detectSingaporePII(content, fileName) {
  // NRIC/FIN: S/T/F/G followed by 7 digits and a letter
  const nricPattern = /\b[STFG]\d{7}[A-Z]\b/i;
  // SingPass identifier patterns
  const singpassPattern = /singpass/i;

  if (nricPattern.test(content)) {
    throw new Error(`Singapore PII (NRIC/FIN) detected in file: ${fileName}. Skipping indexing.`);
  }
  if (singpassPattern.test(content)) {
    throw new Error(`Singapore PII (SingPass identifier) detected in file: ${fileName}. Skipping indexing.`);
  }
}

// Malicious content / prompt injection detection
function detectMaliciousContent(content, fileName) {
  // Check for hidden/invisible characters (beyond what sanitize strips, e.g. zero-width)
  const hiddenCharsPattern = /[\u200B-\u200D\uFEFF\u00AD]/;
  if (hiddenCharsPattern.test(content)) {
    throw new Error(`Hidden/invisible characters detected in file: ${fileName}. Rejecting.`);
  }

  // Check for base64-encoded payloads (long base64 strings)
  const base64Pattern = /(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?/;
  if (base64Pattern.test(content)) {
    throw new Error(`Potential base64-encoded payload detected in file: ${fileName}. Rejecting.`);
  }

  // Check for leetspeak patterns (common substitutions)
  const leetspeakPattern = /(\b\w*[0@][a-z]*\w*\b|\b\w*[1!][a-z]*\w*\b|\b\w*[3][a-z]*\w*\b){3,}/i;
  if (leetspeakPattern.test(content)) {
    throw new Error(`Leetspeak pattern detected in file: ${fileName}. Rejecting.`);
  }

  // Check for binary/shell command patterns
  const shellCommandPattern = /(\/bin\/|\/etc\/|\/usr\/|bash|sh\s+-c|cmd\.exe|powershell|eval\s*\(|exec\s*\(|system\s*\(|`[^`]+`|\$\([^)]+\))/i;
  if (shellCommandPattern.test(content)) {
    throw new Error(`Binary/shell command pattern detected in file: ${fileName}. Rejecting.`);
  }

  // Check for suspicious prompt-injection keywords
  const promptInjectionPattern = /\b(ignore previous instructions|disregard (all |prior |previous )?instructions|you are now|act as|pretend (to be|you are)|jailbreak|override (instructions|system|prompt)|forget (all |your |previous )?instructions|new instructions|system prompt|bypass|do not follow)\b/i;
  if (promptInjectionPattern.test(content)) {
    throw new Error(`Suspicious prompt-injection pattern detected in file: ${fileName}. Rejecting.`);
  }
}

// Validate file path to prevent path traversal
function isValidFilePath(baseDir, filePath) {
  const resolvedBase = path.resolve(baseDir);
  const resolvedFile = path.resolve(filePath);
  return resolvedFile.startsWith(resolvedBase + path.sep) || resolvedFile === resolvedBase;
}

// Validate filename to safe characters only
function isSafeFileName(fileName) {
  return /^[a-zA-Z0-9_\-\.]+$/.test(fileName);
}

const baseDir = "companions";
const fileNames = fs.readdirSync(baseDir);
const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

const langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
    if (fileName.endsWith(".txt")) {
      // Validate filename characters
      if (!isSafeFileName(fileName)) {
        console.warn(`Skipping file with unsafe filename: ${fileName}`);
        return undefined;
      }

      const filePath = path.join(baseDir, fileName);

      // Validate path to prevent path traversal
      if (!isValidFilePath(baseDir, filePath)) {
        console.warn(`Skipping file outside base directory: ${fileName}`);
        return undefined;
      }

      // Check file size
      const stats = fs.statSync(filePath);
      if (stats.size > MAX_FILE_SIZE) {
        console.warn(`Skipping file exceeding size limit: ${fileName}`);
        return undefined;
      }

      let fileContent = fs.readFileSync(filePath, "utf8");

      // Sanitize content
      fileContent = sanitizeContent(fileContent);

      try {
        validateContent(fileContent);
      } catch (e) {
        console.warn(`Skipping file with invalid content (${fileName}): ${e.message}`);
        return undefined;
      }

      // Detect malicious content / prompt injection
      try {
        detectMaliciousContent(fileContent, fileName);
      } catch (e) {
        console.warn(e.message);
        return undefined;
      }

      // Detect Singapore-specific PII
      try {
        detectSingaporePII(fileContent, fileName);
      } catch (e) {
        console.warn(e.message);
        return undefined;
      }

      // Redact PII
      fileContent = redactPII(fileContent);

      // get the last section in the doc for background info
      const lastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
      const splitDocs = await splitter.createDocuments([lastSection]);
      return splitDocs.map((doc) => {
        let pageContent = sanitizeContent(doc.pageContent);
        pageContent = truncateChunk(pageContent);
        pageContent = redactPII(pageContent);
        return new Document({
          metadata: { fileName },
          pageContent,
        });
      });
    }
  })
);

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

await PineconeStore.fromDocuments(
  langchainDocs.flat().filter((doc) => doc !== undefined),
  new HuggingFaceInferenceEmbeddings({ apiKey: process.env.HUGGINGFACEHUB_API_KEY }),
  {
    pineconeIndex,
  }
);