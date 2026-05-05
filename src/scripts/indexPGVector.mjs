// Call embeding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

// External systems accessed by this script:
// 1. Supabase (SUPABASE_URL, SUPABASE_PRIVATE_KEY)
// 2. HuggingFace Inference API (HUGGINGFACEHUB_API_KEY)

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { CharacterTextSplitter } from "langchain/text_splitter";

import fs from "fs";
import path from "path";

dotenv.config({ path: `.env.local` });

const MAX_CONTENT_LENGTH = 100000;

// Sanitize and validate file content: strip null bytes, control characters, excessively long content
function sanitizeFileContent(content) {
  if (typeof content !== "string") return "";
  // Strip null bytes and control characters (except newline, carriage return, tab)
  let sanitized = content.replace(/\x00/g, "").replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Truncate excessively long content
  if (sanitized.length > MAX_CONTENT_LENGTH) {
    sanitized = sanitized.slice(0, MAX_CONTENT_LENGTH);
  }
  return sanitized;
}

// Sanitize against malicious prompt injection, hidden prompts, invisible characters,
// base64-encoded prompts, leetspeak prompts, suspicious instruction patterns, binary/shell commands
function sanitizePromptInjection(content) {
  if (typeof content !== "string") return "";
  // Remove invisible/zero-width characters
  let sanitized = content.replace(/[\u200B-\u200D\uFEFF\u00AD\u2060]/g, "");
  // Remove base64-encoded blocks that look like injected prompts (long base64 strings)
  sanitized = sanitized.replace(/(?:[A-Za-z0-9+/]{40,}={0,2})/g, "[REDACTED_BASE64]");
  // Remove suspicious instruction patterns
  const suspiciousPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /system\s*prompt/gi,
    /you\s+are\s+now/gi,
    /act\s+as\s+(a\s+)?(?:an?\s+)?(?:evil|malicious|unrestricted)/gi,
    /disregard\s+(all\s+)?(previous|prior|above)/gi,
    /forget\s+(all\s+)?(previous|prior|above)/gi,
    /new\s+instructions?:/gi,
    /override\s+(all\s+)?instructions?/gi,
  ];
  for (const pattern of suspiciousPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED]");
  }
  // Remove shell/binary command patterns
  sanitized = sanitized.replace(/(\$\(|\`)[^\)`]*(\)|\`)/g, "[REDACTED_CMD]");
  sanitized = sanitized.replace(/\b(rm\s+-rf|chmod|chown|wget|curl\s+.*\|.*sh|bash\s+-c|sh\s+-c|exec\s+|\/bin\/sh|\/bin\/bash)\b/gi, "[REDACTED_CMD]");
  return sanitized;
}

// Redact PII: SSN, email, phone, home address patterns, credit card numbers
function redactPII(content) {
  if (typeof content !== "string") return "";
  let redacted = content;
  // SSN (US): XXX-XX-XXXX
  redacted = redacted.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Email addresses
  redacted = redacted.replace(/\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g, "[REDACTED_EMAIL]");
  // Phone numbers (various formats)
  redacted = redacted.replace(/(\+?\d[\s\-.]?)?(\(?\d{3}\)?[\s\-.]?)(\d{3}[\s\-.]?\d{4})/g, "[REDACTED_PHONE]");
  // Credit card numbers (16 digits, with or without spaces/dashes)
  redacted = redacted.replace(/\b(?:\d[ \-]?){13,16}\b/g, "[REDACTED_CC]");
  // Home address patterns (basic: number + street name + street type)
  redacted = redacted.replace(/\b\d{1,5}\s+\w+(\s+\w+){0,3}\s+(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b/gi, "[REDACTED_ADDRESS]");
  return redacted;
}

// Singapore-specific PII detection: NRIC/FIN, SingPass, full names, personal emails
function detectSingaporePII(content, fileName) {
  if (typeof content !== "string") return;
  // NRIC/FIN: S/T/F/G followed by 7 digits and a letter
  const nricPattern = /\b[STFG]\d{7}[A-Z]\b/gi;
  if (nricPattern.test(content)) {
    throw new Error(`Singapore PII detected (NRIC/FIN) in file: ${fileName}. Skipping.`);
  }
  // SingPass identifier patterns
  const singpassPattern = /singpass/gi;
  if (singpassPattern.test(content)) {
    throw new Error(`Singapore PII detected (SingPass reference) in file: ${fileName}. Skipping.`);
  }
  // Personal email addresses (already handled by redactPII, but check for remaining after redaction)
  const emailPattern = /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g;
  if (emailPattern.test(content)) {
    throw new Error(`Singapore PII detected (personal email) in file: ${fileName}. Skipping.`);
  }
}

// Validate and sanitize LLM output / document pageContent for dynamic code execution primitives
function sanitizeLLMOutput(pageContent) {
  if (typeof pageContent !== "string") return "";
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /new\s+Function\s*\(/gi,
    /setTimeout\s*\(\s*["'`]/gi,
    /setInterval\s*\(\s*["'`]/gi,
    /\bimportScripts\s*\(/gi,
    /document\.write\s*\(/gi,
    /\.innerHTML\s*=/gi,
    /\bprocess\.binding\s*\(/gi,
    /require\s*\(\s*["'`]child_process["'`]\s*\)/gi,
  ];
  let sanitized = pageContent;
  for (const pattern of dangerousPatterns) {
    if (pattern.test(sanitized)) {
      sanitized = sanitized.replace(pattern, "[REDACTED_CODE]");
    }
  }
  return sanitized;
}

// Validate final document pageContent
function isValidDocument(doc) {
  if (!doc || typeof doc.pageContent !== "string") return false;
  const content = doc.pageContent.trim();
  if (content.length === 0) return false;
  if (content.length > MAX_CONTENT_LENGTH) return false;
  return true;
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
      let fileContent = fs.readFileSync(filePath, "utf8");

      // Sanitize and validate file content before processing
      fileContent = sanitizeFileContent(fileContent);
      fileContent = sanitizePromptInjection(fileContent);
      fileContent = redactPII(fileContent);

      // Detect Singapore-specific PII; skip file if found
      try {
        detectSingaporePII(fileContent, fileName);
      } catch (err) {
        console.error(err.message);
        return [];
      }

      const lastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
      const splitDocs = await splitter.createDocuments([lastSection]);
      return splitDocs.map((doc) => {
        // Sanitize LLM output / document content for dynamic code execution primitives
        const sanitizedContent = sanitizeLLMOutput(doc.pageContent);
        return new Document({
          metadata: { fileName },
          pageContent: sanitizedContent,
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

// Filter and validate all documents before passing to the vector store
const validDocs = langchainDocs.flat().filter((doc) => doc !== undefined && isValidDocument(doc));

await SupabaseVectorStore.fromDocuments(
  validDocs,
  new HuggingFaceInferenceEmbeddings({ apiKey: process.env.HUGGINGFACEHUB_API_KEY }),
  {
    client,
    tableName: "documents",
  }
);