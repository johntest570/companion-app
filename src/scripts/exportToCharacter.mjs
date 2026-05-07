import { Redis } from "@upstash/redis";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import { ChatAnthropic } from "langchain/chat_models/anthropic";

import dotenv from "dotenv";
import fs from "fs/promises";
import path from "path";
dotenv.config({ path: `.env.local` });

// ── Credentials block ────────────────────────────────────────────────────────
const UPSTASH_REDIS_REST_URL = process.env.UPSTASH_REDIS_REST_URL;
const UPSTASH_REDIS_REST_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const EXPORT_SECRET = process.env.EXPORT_SECRET;

if (!UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN) {
  throw new Error("Missing required Redis credentials in environment.");
}
if (!ANTHROPIC_API_KEY) {
  throw new Error("Missing required ANTHROPIC_API_KEY in environment.");
}
if (!EXPORT_SECRET) {
  throw new Error("Missing required EXPORT_SECRET in environment.");
}
// ─────────────────────────────────────────────────────────────────────────────

const COMPANION_NAME = process.argv[2];
const MODEL_NAME = process.argv[3];
const PROVIDED_TOKEN = process.argv[4];
const USER_ID = process.argv[5];

if (!COMPANION_NAME || !MODEL_NAME || !PROVIDED_TOKEN || !USER_ID) {
  throw new Error(
    "**Usage**: npm run generate-character <COMPANION_NAME> <MODEL_NAME> <SECRET_TOKEN> <USER_ID>"
  );
}

// ── Authentication check ─────────────────────────────────────────────────────
if (PROVIDED_TOKEN !== EXPORT_SECRET) {
  throw new Error("Authentication failed: invalid secret token.");
}
// ─────────────────────────────────────────────────────────────────────────────

// ── COMPANION_NAME validation (path traversal / injection prevention) ─────────
const SAFE_NAME_RE = /^[a-zA-Z0-9_-]+$/;
if (!SAFE_NAME_RE.test(COMPANION_NAME)) {
  throw new Error(
    "Invalid COMPANION_NAME: only alphanumeric characters, hyphens, and underscores are allowed."
  );
}
// ─────────────────────────────────────────────────────────────────────────────

// ── Explicit tool / question allow-list ──────────────────────────────────────
const ALLOWED_QUESTIONS = [
  `Greeting: What would ${COMPANION_NAME} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
  `Long Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
];
// ─────────────────────────────────────────────────────────────────────────────

// ── Sanitization helpers ─────────────────────────────────────────────────────

/**
 * Strip non-printable / control characters and common prompt-injection patterns.
 */
function sanitizeInput(text) {
  if (typeof text !== "string") return "";
  // Remove non-printable / control characters (except newline and tab)
  let sanitized = text.replace(/[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]/g, "");
  // Strip prompt-injection patterns
  sanitized = sanitized.replace(
    /\b(ignore previous instructions?|disregard (all )?previous|you are now|act as|system prompt|<\/?s>|<\/?system>)\b/gi,
    "[REDACTED]"
  );
  return sanitized;
}

/**
 * Detect and strip hidden/invisible characters, base64-encoded payloads,
 * leetspeak, shell/binary commands, and other suspicious patterns from
 * file-sourced content.
 */
function sanitizeFileContent(text) {
  if (typeof text !== "string") return "";
  // Remove zero-width and other invisible Unicode characters
  let sanitized = text.replace(
    /[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u00A0]/g,
    ""
  );
  // Remove non-printable control characters (except newline and tab)
  sanitized = sanitized.replace(/[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]/g, "");
  // Detect and redact base64-encoded blobs (long base64 strings)
  sanitized = sanitized.replace(
    /(?:[A-Za-z0-9+/]{40,}={0,2})/g,
    "[BASE64_REDACTED]"
  );
  // Redact shell command patterns
  sanitized = sanitized.replace(
    /\b(bash|sh|cmd|powershell|exec|system|popen|subprocess|os\.system|eval|Function\s*\(|new\s+Function)\b/gi,
    "[CMD_REDACTED]"
  );
  // Redact prompt injection patterns
  sanitized = sanitized.replace(
    /\b(ignore previous instructions?|disregard (all )?previous|you are now|act as|system prompt|<\/?s>|<\/?system>)\b/gi,
    "[REDACTED]"
  );
  return sanitized;
}

/**
 * Sanitize LLM output: reject / strip dynamic code execution primitives.
 */
function sanitizeLLMOutput(text) {
  if (typeof text !== "string") return "";
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*["'`]/gi,
    /\bsetInterval\s*\(\s*["'`]/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bprocess\.binding\s*\(/gi,
    /\bchild_process/gi,
    /\bspawn\s*\(/gi,
    /\bexecSync\s*\(/gi,
    /\bexecFile\s*\(/gi,
  ];
  let sanitized = text;
  for (const pattern of dangerousPatterns) {
    sanitized = sanitized.replace(pattern, "[BLOCKED]");
  }
  return sanitized;
}

// ─────────────────────────────────────────────────────────────────────────────

// ── Restricted output directory ───────────────────────────────────────────────
const OUTPUT_DIR = path.resolve("output");
await fs.mkdir(OUTPUT_DIR, { recursive: true });
// ─────────────────────────────────────────────────────────────────────────────

// ── Read and sanitize companion file ─────────────────────────────────────────
const companionsDir = path.resolve("companions");
const companionFilePath = path.join(companionsDir, COMPANION_NAME + ".txt");
// Ensure the resolved path stays within the companions directory
if (!companionFilePath.startsWith(companionsDir + path.sep)) {
  throw new Error("Path traversal detected in COMPANION_NAME.");
}

const data = await fs.readFile(companionFilePath, "utf8");
const presplit = data.split("###ENDPREAMBLE###");
const preamble = sanitizeFileContent(sanitizeInput(presplit[0]));
const seedsplit = presplit[1].split("###ENDSEED###");
const seedChat = sanitizeFileContent(sanitizeInput(seedsplit[0]));
const backgroundStory = sanitizeFileContent(sanitizeInput(seedsplit[1]));
console.log(preamble, backgroundStory);
// ─────────────────────────────────────────────────────────────────────────────

const history = new Redis({
  url: UPSTASH_REDIS_REST_URL,
  token: UPSTASH_REDIS_REST_TOKEN,
});

const upstashChatHistory = await history.zrange(
  `${COMPANION_NAME}-${MODEL_NAME}-${USER_ID}`,
  0,
  Date.now(),
  {
    byScore: true,
  }
);
const recentChat = upstashChatHistory
  .slice(-30)
  .map((entry) => sanitizeInput(String(entry)));

const model = new ChatAnthropic({
  modelName: "claude-2",
  anthropicApiKey: ANTHROPIC_API_KEY,
});
model.verbose = true;

const sanitizedCompanionName = sanitizeInput(COMPANION_NAME);

const chainPrompt = PromptTemplate.fromTemplate(`
  ### Background Story: 
  ${preamble}
  
  ${backgroundStory}

  ### Chat history: 
  ${seedChat}

  ...
  ${recentChat.join("\n")}

  
  Above is someone whose name is ${sanitizedCompanionName}'s story and their chat history with a human. Output answer to the following question. Return only the answer itself 
  
  {question}`);

const chain = new LLMChain({
  llm: model,
  prompt: chainPrompt,
});

const questions = ALLOWED_QUESTIONS;

// ── LLM interaction log ───────────────────────────────────────────────────────
const llmInteractionLog = [];
// ─────────────────────────────────────────────────────────────────────────────

const results = await Promise.all(
  questions.map(async (question) => {
    // Validate question against allow-list
    if (!ALLOWED_QUESTIONS.includes(question)) {
      throw new Error(`Question not in allow-list: ${question}`);
    }
    const sanitizedQuestion = sanitizeInput(question);
    const timestamp = new Date().toISOString();
    let responseText = null;
    let errorMsg = null;
    try {
      const result = await chain.call({ question: sanitizedQuestion });
      responseText = sanitizeLLMOutput(result.text);
      llmInteractionLog.push({
        timestamp,
        input: sanitizedQuestion,
        output: responseText,
      });
      return { text: responseText };
    } catch (error) {
      errorMsg = error.message || String(error);
      llmInteractionLog.push({
        timestamp,
        input: sanitizedQuestion,
        output: null,
        error: errorMsg,
      });
      console.error(error);
      throw error;
    }
  })
);

// Write LLM interaction log
const logFilePath = path.join(OUTPUT_DIR, `llm_interactions_${Date.now()}.json`);
await fs.writeFile(logFilePath, JSON.stringify(llmInteractionLog, null, 2), "utf8");

let output = "";
for (let i = 0; i < questions.length; i++) {
  output += `*****${questions[i]}*****\n${results[i].text}\n\n`;
}
output += `Definition (Advanced)\n${recentChat.join("\n")}`;

// Write output files to restricted output directory only; do not write raw PII chat history
await fs.writeFile(
  path.join(OUTPUT_DIR, `${COMPANION_NAME}_character_ai_data.txt`),
  output,
  "utf8"
);