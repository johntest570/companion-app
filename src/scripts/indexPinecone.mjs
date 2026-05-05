// Major ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pinecone
import { PineconeClient } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { Document } from "langchain/document";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";

// External systems accessed by this script:
// 1. Pinecone (PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX)
// 2. HuggingFace Inference API (HUGGINGFACEHUB_API_KEY)

dotenv.config({ path: `.env.local` });

const MAX_CONTENT_LENGTH = 100000;

function sanitizeContent(content) {
  // Strip null bytes and non-printable characters (except common whitespace)
  let sanitized = content.replace(/\0/g, "");
  sanitized = sanitized.replace(/[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]/g, "");
  // Trim whitespace
  sanitized = sanitized.trim();
  // Enforce maximum length
  if (sanitized.length > MAX_CONTENT_LENGTH) {
    sanitized = sanitized.slice(0, MAX_CONTENT_LENGTH);
  }
  // Reject empty content
  if (sanitized.length === 0) {
    throw new Error("Content is empty after sanitization.");
  }
  return sanitized;
}

function detectMaliciousContent(content) {
  // Check for hidden/invisible characters
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content)) {
    throw new Error("File contains hidden or invisible characters.");
  }
  // Check for base64-encoded prompts (long base64 strings)
  if (/[A-Za-z0-9+/]{100,}={0,2}/.test(content)) {
    throw new Error("File contains potential base64-encoded content.");
  }
  // Check for leetspeak patterns
  if (/(\b\w*[0-9]\w*[0-9]\w*[0-9]\w*\b.*){3,}/i.test(content)) {
    throw new Error("File contains potential leetspeak patterns.");
  }
  // Check for suspicious AI instruction patterns
  const aiInstructionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+(a\s+)?(\w+\s+)?AI/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /system\s*:\s*you\s+are/i,
    /\[INST\]/i,
    /<<SYS>>/i,
    /###\s*instruction/i,
    /prompt\s*injection/i,
    /jailbreak/i,
  ];
  for (const pattern of aiInstructionPatterns) {
    if (pattern.test(content)) {
      throw new Error("File contains suspicious AI instruction patterns.");
    }
  }
  // Check for binary/shell commands
  const shellCommandPatterns = [
    /\b(bash|sh|zsh|cmd|powershell|exec|eval|system|popen)\s*[\(\[]/i,
    /\b(rm\s+-rf|chmod|chown|sudo|wget|curl)\b/i,
    /\$\(.*\)/,
    /`[^`]+`/,
  ];
  for (const pattern of shellCommandPatterns) {
    if (pattern.test(content)) {
      throw new Error("File contains potential shell commands.");
    }
  }
}

function redactPII(content) {
  // Redact SSNs (e.g., 123-45-6789)
  let redacted = content.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Redact email addresses
  redacted = redacted.replace(
    /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
    "[REDACTED_EMAIL]"
  );
  // Redact phone numbers (various formats)
  redacted = redacted.replace(
    /(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})/g,
    "[REDACTED_PHONE]"
  );
  // Redact credit card numbers
  redacted = redacted.replace(
    /\b(?:\d[ -]?){13,16}\b/g,
    "[REDACTED_CREDIT_CARD]"
  );
  // Redact dates of birth (common formats)
  redacted = redacted.replace(
    /\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](19|20)\d{2}\b/g,
    "[REDACTED_DOB]"
  );
  // Redact medical record numbers (MRN patterns)
  redacted = redacted.replace(
    /\b(MRN|Medical Record(?: Number)?)[:\s#]*\d+\b/gi,
    "[REDACTED_MRN]"
  );
  // Redact home addresses (basic pattern: number + street name + street type)
  redacted = redacted.replace(
    /\b\d+\s+[A-Za-z0-9\s,\.]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b/gi,
    "[REDACTED_ADDRESS]"
  );
  return redacted;
}

function detectSingaporePII(content, fileName) {
  // NRIC/FIN numbers (e.g., S1234567A, T0123456B, F1234567C, G1234567D)
  if (/\b[STFG]\d{7}[A-Z]\b/.test(content)) {
    throw new Error(
      `File ${fileName} contains Singapore NRIC/FIN number. Skipping.`
    );
  }
  // Singapore passport numbers (e.g., E1234567A)
  if (/\b[A-Z]\d{7}[A-Z]\b/.test(content)) {
    throw new Error(
      `File ${fileName} contains potential Singapore passport number. Skipping.`
    );
  }
  // Singapore phone numbers (+65 XXXX XXXX or 8/9 XXXX XXXX)
  if (/(\+65[\s-]?)?[89]\d{3}[\s-]?\d{4}\b/.test(content)) {
    throw new Error(
      `File ${fileName} contains Singapore phone number. Skipping.`
    );
  }
  // Health record indicators
  if (
    /\b(patient|diagnosis|prescription|medical record|health record|clinical)\b/i.test(
      content
    )
  ) {
    throw new Error(
      `File ${fileName} contains health record indicators. Skipping.`
    );
  }
  // Full name patterns (common: two or more capitalized words)
  if (/\b([A-Z][a-z]+\s){2,}[A-Z][a-z]+\b/.test(content)) {
    throw new Error(
      `File ${fileName} contains potential full name (PII). Skipping.`
    );
  }
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
      const filePath = path.join("companions", fileName);
      const fileContent = fs.readFileSync(filePath, "utf8");
      // get the last section in the doc for background info
      let lastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];

      // Detect malicious content in file
      try {
        detectMaliciousContent(lastSection);
      } catch (err) {
        console.error(`Skipping file ${fileName}: ${err.message}`);
        return undefined;
      }

      // Detect Singapore PII
      try {
        detectSingaporePII(lastSection, fileName);
      } catch (err) {
        console.error(`Skipping file ${fileName}: ${err.message}`);
        return undefined;
      }

      // Redact PII from content
      lastSection = redactPII(lastSection);

      // Sanitize and validate content
      try {
        lastSection = sanitizeContent(lastSection);
      } catch (err) {
        console.error(`Skipping file ${fileName}: ${err.message}`);
        return undefined;
      }

      const splitDocs = await splitter.createDocuments([lastSection]);
      return splitDocs.map((doc) => {
        return new Document({
          metadata: { fileName },
          pageContent: doc.pageContent,
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
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
  }),
  {
    pineconeIndex,
  }
);