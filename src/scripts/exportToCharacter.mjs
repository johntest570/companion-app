import { Redis } from "@upstash/redis";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import { ChatAnthropic } from "langchain/chat_models/anthropic";

import dotenv from "dotenv";
import fs from "fs/promises";
dotenv.config({ path: `.env.local` });

// Credentialed external systems (2 total — compliant with ≤3 policy):
//   1. Upstash Redis  (UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN)
//   2. Anthropic Claude (ANTHROPIC_API_KEY)

const COMPANION_NAME = process.argv[2];
const MODEL_NAME = process.argv[3];
const USER_ID = process.argv[4];

if (!!!COMPANION_NAME || !!!MODEL_NAME || !!!USER_ID) {
  throw new Error(
    "**Usage**: npm run export-to-character <COMPANION_NAME> <MODEL_NAME> <USER_ID>"
  );
}

// ---------------------------------------------------------------------------
// Sanitization helpers
// ---------------------------------------------------------------------------

/**
 * Strips prompt-injection characters and enforces a length limit on
 * user-controlled scalar values (COMPANION_NAME, MODEL_NAME, USER_ID).
 */
function sanitizeInput(value, maxLength = 256) {
  if (typeof value !== "string") return "";
  // Remove null bytes, control characters (except newline/tab), and common
  // prompt-injection delimiters.
  let sanitized = value
    .replace(/\0/g, "")
    .replace(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/g, "")
    .replace(/[<>{}|\\^`]/g, "")
    .trim();
  return sanitized.slice(0, maxLength);
}

/**
 * Strips hidden/invisible Unicode characters, base64-encoded blobs,
 * binary/shell command patterns, leetspeak-obfuscated prompt injections,
 * and explicit prompt-override phrases from file content.
 */
function sanitizeFileContent(text) {
  if (typeof text !== "string") return "";

  // Remove invisible / zero-width Unicode characters
  let sanitized = text.replace(
    /[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u00A0]/g,
    ""
  );

  // Remove base64-encoded blobs (long runs of base64 chars)
  sanitized = sanitized.replace(/[A-Za-z0-9+/]{100,}={0,2}/g, "[REDACTED_B64]");

  // Remove shell/binary command patterns
  sanitized = sanitized.replace(
    /(\b)(bash|sh|cmd|powershell|exec|system|popen|subprocess|eval|Function)\s*[\(\[`]/gi,
    "[REDACTED_CMD]"
  );

  // Remove explicit prompt-override phrases
  const overridePhrases = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /you\s+are\s+now\s+/gi,
    /new\s+instructions?:/gi,
    /system\s*prompt:/gi,
    /###\s*instruction/gi,
  ];
  for (const pattern of overridePhrases) {
    sanitized = sanitized.replace(pattern, "[REDACTED_INJECTION]");
  }

  // Remove leetspeak-obfuscated variants (simple pass)
  sanitized = sanitized.replace(
    /[1!][Gg][Nn][Oo][Rr][Ee]/g,
    "[REDACTED_LEET]"
  );

  return sanitized;
}

/**
 * Redacts common PII patterns from text.
 */
function redactPII(text) {
  if (typeof text !== "string") return "";

  let redacted = text;

  // Email addresses
  redacted = redacted.replace(
    /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g,
    "[REDACTED_EMAIL]"
  );

  // Phone numbers (various formats)
  redacted = redacted.replace(
    /(\+?\d[\d\s\-().]{7,}\d)/g,
    "[REDACTED_PHONE]"
  );

  // SSNs (US)
  redacted = redacted.replace(
    /\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/g,
    "[REDACTED_SSN]"
  );

  // Credit card numbers
  redacted = redacted.replace(
    /\b(?:\d[ \-]?){13,16}\b/g,
    "[REDACTED_CC]"
  );

  // IPv4 addresses
  redacted = redacted.replace(
    /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g,
    "[REDACTED_IP]"
  );

  // Singapore NRIC/FIN (e.g. S1234567A, T9876543Z, F1234567X, G1234567P)
  redacted = redacted.replace(
    /\b[STFG]\d{7}[A-Z]\b/g,
    "[REDACTED_NRIC]"
  );

  // Singapore SingPass identifiers (common prefix patterns)
  redacted = redacted.replace(
    /\bSingPass[\s:]*\S+/gi,
    "[REDACTED_SINGPASS]"
  );

  return redacted;
}

/**
 * Detects Singapore-specific PII in file content and throws if found.
 */
function detectSingaporePII(text) {
  const patterns = [
    { name: "NRIC/FIN", regex: /\b[STFG]\d{7}[A-Z]\b/ },
    { name: "SingPass", regex: /\bSingPass[\s:]*\S+/i },
    {
      name: "SG_DOB",
      regex: /\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-](19|20)\d{2}\b/,
    },
    {
      name: "SG_ADDRESS",
      regex: /\b(Blk|Block|#\d{2}-\d{2,4}|Singapore\s+\d{6})\b/i,
    },
  ];

  for (const { name, regex } of patterns) {
    if (regex.test(text)) {
      throw new Error(
        `File contains Singapore PII (${name}). Processing aborted.`
      );
    }
  }
}

/**
 * Validates LLM output by removing dynamic code execution primitives.
 */
function sanitizeLLMOutput(text) {
  if (typeof text !== "string") return "";

  let sanitized = text;

  // Remove eval, exec, subprocess, Function constructor, etc.
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bsubprocess\s*\./gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*["'`]/gi,
    /\bsetInterval\s*\(\s*["'`]/gi,
    /\bimport\s*\(\s*["'`]/gi,
    /\brequire\s*\(\s*["'`]/gi,
    /\bchild_process\b/gi,
    /\bspawnSync\s*\(/gi,
    /\bexecSync\s*\(/gi,
    /\bexecFileSync\s*\(/gi,
    /\bvm\.runInThisContext\s*\(/gi,
    /\bvm\.runInNewContext\s*\(/gi,
  ];

  for (const pattern of dangerousPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED_CODE_EXEC]");
  }

  return sanitized;
}

// ---------------------------------------------------------------------------
// Logging helper
// ---------------------------------------------------------------------------

const LOG_FILE = `llm_interactions_${Date.now()}.log`;

async function logLLMInteraction(question, response) {
  const entry = JSON.stringify({
    timestamp: new Date().toISOString(),
    input: question,
    output: response,
  });
  await fs.appendFile(LOG_FILE, entry + "\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const sanitizedCompanionName = sanitizeInput(COMPANION_NAME);
const sanitizedModelName = sanitizeInput(MODEL_NAME);
const sanitizedUserId = sanitizeInput(USER_ID);

const rawData = await fs.readFile(
  "companions/" + sanitizedCompanionName + ".txt",
  "utf8"
);

// Detect Singapore PII before any further processing
detectSingaporePII(rawData);

// Sanitize file content for malicious prompt injections
const data = sanitizeFileContent(rawData);

const presplit = data.split("###ENDPREAMBLE###");
const rawPreamble = presplit[0];
const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
const rawSeedChat = seedsplit[0];
const rawBackgroundStory = seedsplit[1];

// Redact PII from file-sourced content before sending to LLM
const preamble = redactPII(rawPreamble);
const seedChat = redactPII(rawSeedChat);
const backgroundStory = redactPII(rawBackgroundStory);

console.log(preamble, backgroundStory);

const history = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

const upstashChatHistory = await history.zrange(
  `${sanitizedCompanionName}-${sanitizedModelName}-${sanitizedUserId}`,
  0,
  Date.now(),
  {
    byScore: true,
  }
);
const recentChatRaw = upstashChatHistory.slice(-30);

// Redact PII from chat history before sending to LLM; keep raw for file output
const recentChat = recentChatRaw.map((msg) =>
  redactPII(sanitizeInput(String(msg), 2048))
);

const model = new ChatAnthropic({
  modelName: "claude-2",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
});
model.verbose = true;

const chainPrompt = PromptTemplate.fromTemplate(`
  ### Background Story: 
  ${preamble}
  
  ${backgroundStory}

  ### Chat history: 
  ${seedChat}

  ...
  ${recentChat}

  
  Above is someone whose name is ${sanitizedCompanionName}'s story and their chat history with a human. Output answer to the following question. Return only the answer itself 
  
  {question}`);

const chain = new LLMChain({
  llm: model,
  prompt: chainPrompt,
});

const questions = [
  `Greeting: What would ${sanitizedCompanionName} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${sanitizedCompanionName} describe themselves?`,
  `Long Description: In a few sentences, how would ${sanitizedCompanionName} describe themselves?`,
];

// Sanitize question strings before passing to chain
const sanitizedQuestions = questions.map((q) => sanitizeInput(q, 512));

const results = await Promise.all(
  sanitizedQuestions.map(async (question) => {
    try {
      // Log input before calling LLM
      await logLLMInteraction(question, null);

      const result = await chain.call({ question });

      // Sanitize LLM output
      if (result && result.text) {
        result.text = sanitizeLLMOutput(result.text);
      }

      // Log output after receiving response
      await logLLMInteraction(question, result ? result.text : null);

      return result;
    } catch (error) {
      console.error(error);
      await logLLMInteraction(question, `ERROR: ${error.message}`);
    }
  })
);

let output = "";
for (let i = 0; i < sanitizedQuestions.length; i++) {
  const sanitizedText =
    results[i] && results[i].text
      ? sanitizeLLMOutput(results[i].text)
      : "";
  output += `*****${sanitizedQuestions[i]}*****\n${sanitizedText}\n\n`;
}
output += `Definition (Advanced)\n${recentChatRaw.join("\n")}`;

await fs.writeFile(`${sanitizedCompanionName}_chat_history.txt`, upstashChatHistory);
await fs.writeFile(`${sanitizedCompanionName}_character_ai_data.txt`, output);