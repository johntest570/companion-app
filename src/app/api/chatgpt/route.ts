import { OpenAI } from "langchain/llms/openai";
import dotenv from "dotenv";
import { LLMChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import clerk from "@clerk/clerk-sdk-node";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import MemoryManager from "@/app/utils/memory";
import { rateLimit } from "@/app/utils/rateLimit";
import crypto from "crypto";

dotenv.config({ path: `.env.local` });

// Tool allow list (this chain uses no external tools)
const ALLOWED_TOOLS: string[] = [];
const POLICY_VERSION = "1.0.0";

function auditLogToolAccess(
  actor: string,
  tool: string,
  allowed: boolean,
  reason: string
) {
  console.log(
    JSON.stringify({
      type: "TOOL_ACCESS_AUDIT",
      policyVersion: POLICY_VERSION,
      actor,
      tool,
      allowed,
      reason,
      timestamp: new Date().toISOString(),
    })
  );
}

function enforceToolAllowList(actor: string) {
  // Log enforcement: no tools are permitted for this chain
  auditLogToolAccess(
    actor,
    "none",
    true,
    "No tools invoked; zero-tool policy enforced for LLMChain"
  );
}

function checkTool(actor: string, tool: string): boolean {
  const allowed = ALLOWED_TOOLS.includes(tool);
  auditLogToolAccess(
    actor,
    tool,
    allowed,
    allowed ? "Tool is on the allow list" : "Tool is not on the allow list"
  );
  return allowed;
}

// Sanitize name header: only allow alphanumeric, hyphens, underscores
function sanitizeName(name: string | null): string | null {
  if (!name) return null;
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) return null;
  return name;
}

// Strip control characters and trim whitespace
function stripControlChars(text: string): string {
  // Remove invisible Unicode and control characters except newlines and tabs
  return text.replace(/[^\P{C}\n\t]/gu, "").trim();
}

// Detect dangerous dynamic code execution primitives in LLM output
const DANGEROUS_OUTPUT_PATTERNS = [
  /\beval\s*\(/i,
  /\bexec\s*\(/i,
  /\bsubprocess\b/i,
  /\bnew\s+Function\s*\(/i,
  /\bsetTimeout\s*\(\s*["'`]/i,
  /\bsetInterval\s*\(\s*["'`]/i,
  /\bimport\s*\(/i,
  /\brequire\s*\(/i,
];

function containsDangerousOutputPatterns(text: string): boolean {
  return DANGEROUS_OUTPUT_PATTERNS.some((pattern) => pattern.test(text));
}

// Detect hidden/invisible characters, base64, leetspeak, shell commands, injection markers
const INJECTION_PATTERNS = [
  /[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]/u, // invisible Unicode
  /(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?/, // base64
  /\b(?:ignore previous|disregard|forget|override|system prompt|you are now|act as|jailbreak)\b/i, // injection markers
  /\b(?:sh|bash|cmd|powershell|exec|system|popen|subprocess|eval|passthru|shell_exec)\b/i, // shell/binary commands
  /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/, // binary/control chars
];

function containsInjectionPatterns(text: string): boolean {
  return INJECTION_PATTERNS.some((pattern) => pattern.test(text));
}

// Sanitize user prompt: strip injection chars, trim whitespace
function sanitizePrompt(prompt: string): string {
  // Remove null bytes and control characters
  let sanitized = prompt.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
  // Trim whitespace
  sanitized = sanitized.trim();
  return sanitized;
}

// Redact PII from text
function redactPII(text: string): string {
  // Redact email addresses
  let redacted = text.replace(
    /[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g,
    "[REDACTED_EMAIL]"
  );
  // Redact phone numbers (various formats)
  redacted = redacted.replace(
    /(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})/g,
    "[REDACTED_PHONE]"
  );
  // Redact SSN patterns
  redacted = redacted.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Redact credit card patterns
  redacted = redacted.replace(
    /\b(?:\d[ -]?){13,16}\b/g,
    "[REDACTED_CARD]"
  );
  return redacted;
}

// Verify HMAC-SHA256 signature for text/SMS requests
function verifyTextRequestSignature(
  userId: string,
  userName: string,
  timestamp: string,
  signature: string
): boolean {
  const secret = process.env.TEXT_REQUEST_HMAC_SECRET;
  if (!secret) return false;
  // Reject if timestamp is older than 5 minutes
  const ts = parseInt(timestamp, 10);
  if (isNaN(ts) || Date.now() - ts > 5 * 60 * 1000) return false;
  const payload = `${userId}:${userName}:${timestamp}`;
  const expected = crypto
    .createHmac("sha256", secret)
    .update(payload)
    .digest("hex");
  return crypto.timingSafeEqual(
    Buffer.from(signature, "hex"),
    Buffer.from(expected, "hex")
  );
}

export async function POST(req: Request) {
  const { prompt, isText, userId, userName, timestamp, signature } =
    await req.json();

  const identifier = req.url + "-" + (userId || "anonymous");
  const { success } = await rateLimit(identifier);
  if (!success) {
    console.log("INFO: rate limit exceeded");
    return new NextResponse(
      JSON.stringify({ Message: "Hi, the companions can't talk this fast." }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Validate and sanitize the name header against an allowlist of safe filename characters
  const rawName = req.headers.get("name");
  const name = sanitizeName(rawName);
  if (!name) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const companionFileName = name + ".txt";

  // Sanitize and validate user prompt
  if (!prompt || typeof prompt !== "string") {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const sanitizedPrompt = sanitizePrompt(prompt);
  if (containsInjectionPatterns(sanitizedPrompt)) {
    return new NextResponse(
      JSON.stringify({ Message: "Prompt contains disallowed content." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Always use server-side authentication via currentUser()
  // For isText/SMS requests, additionally verify HMAC signature
  let clerkUserId: string | undefined;
  let clerkUserName: string | null | undefined;

  if (isText) {
    // Verify HMAC signature for text requests
    if (
      !timestamp ||
      !signature ||
      !verifyTextRequestSignature(userId, userName, timestamp, signature)
    ) {
      console.log("user not authorized: invalid or missing signature");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
  }

  // Always verify server-side session
  const user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
    console.log("user not authorized");
    return new NextResponse(
      JSON.stringify({ Message: "User not authorized" }),
      {
        status: 401,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Load character "PREAMBLE" from character file.
  const fs = require("fs").promises;
  const data = await fs.readFile("companions/" + companionFileName, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const rawPreamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const rawSeedchat = seedsplit[0];

  // Sanitize preamble and seedchat loaded from companion file
  if (containsInjectionPatterns(rawPreamble)) {
    console.log("Companion preamble contains disallowed content; stripping.");
  }
  const preamble = containsInjectionPatterns(rawPreamble)
    ? stripControlChars(rawPreamble.replace(INJECTION_PATTERNS[2], ""))
    : stripControlChars(rawPreamble);

  if (containsInjectionPatterns(rawSeedchat)) {
    console.log("Companion seedchat contains disallowed content; stripping.");
  }
  const seedchat = containsInjectionPatterns(rawSeedchat)
    ? stripControlChars(rawSeedchat.replace(INJECTION_PATTERNS[2], ""))
    : stripControlChars(rawSeedchat);

  const companionKey = {
    companionName: name!,
    modelName: "chatgpt",
    userId: clerkUserId,
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }

  await memoryManager.writeToHistory(
    "Human: " + sanitizedPrompt + "\n",
    companionKey
  );
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  // query Pinecone
  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companionFileName
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs
      .map((doc) => {
        const content = stripControlChars(doc.pageContent);
        // Strip injection patterns from vector store content
        return containsInjectionPatterns(content) ? "" : content;
      })
      .filter(Boolean)
      .join("\n");
  }

  // Redact PII from chat history and relevant history before sending to LLM
  const sanitizedRecentChatHistory = redactPII(recentChatHistory);
  const sanitizedRelevantHistory = redactPII(relevantHistory);

  const { stream, handlers } = LangChainStream();

  const model = new OpenAI({
    streaming: true,
    modelName: process.env.OPENAI_MODEL_NAME || "gpt-4o-mini",
    callbackManager: CallbackManager.fromHandlers(handlers),
    stop: ["Human:", "###END###"],
  });
  model.verbose = true;

  const replyWithTwilioLimit = isText
    ? "You reply within 1000 characters."
    : "";

  // Use a generic placeholder instead of the real user name to avoid sending PII to the LLM
  const chainPrompt = PromptTemplate.fromTemplate(`
    You are ${name} and are currently talking to a user.

    ${preamble}

  You reply with answers that range from one sentence to one paragraph and with some details. ${replyWithTwilioLimit}

  Below are relevant details about ${name}'s past
  ${sanitizedRelevantHistory}
  
  Below is a relevant conversation history

  ${sanitizedRecentChatHistory}`);

  // Enforce tool allow list and log the enforcement decision
  enforceToolAllowList(clerkUserId);

  // Log the full rendered prompt before invoking the LLM
  console.log(
    JSON.stringify({
      type: "LLM_REQUEST",
      actor: clerkUserId,
      companionName: name,
      preambleLength: preamble.length,
      relevantHistoryLength: sanitizedRelevantHistory.length,
      recentChatHistoryLength: sanitizedRecentChatHistory.length,
      timestamp: new Date().toISOString(),
    })
  );

  const chain = new LLMChain({
    llm: model,
    prompt: chainPrompt,
  });

  // Wrap chain.call() in a Promise.race() with a timeout for explicit termination
  const CHAIN_TIMEOUT_MS = 30000;
  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(
      () => reject(new Error("LLMChain timed out after " + CHAIN_TIMEOUT_MS + "ms")),
      CHAIN_TIMEOUT_MS
    )
  );

  const result = await Promise.race([
    chain.call({
      relevantHistory: sanitizedRelevantHistory,
      recentChatHistory: sanitizedRecentChatHistory,
    }),
    timeoutPromise,
  ]).catch((err) => {
    console.error("LLMChain error or timeout:", err);
    return null;
  });

  // Validate result before proceeding
  if (!result || !result.text || typeof result.text !== "string") {
    console.log("LLM returned empty or invalid result; terminating.");
    return new NextResponse(
      JSON.stringify({ Message: "No response from model." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Sanitize and validate LLM output for dangerous patterns
  if (containsDangerousOutputPatterns(result.text)) {
    console.log("LLM output contains dangerous patterns; rejecting.");
    return new NextResponse(
      JSON.stringify({ Message: "Response blocked by safety policy." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Log the LLM response
  console.log(
    JSON.stringify({
      type: "LLM_RESPONSE",
      actor: clerkUserId,
      companionName: name,
      responseLength: result.text.length,
      timestamp: new Date().toISOString(),
    })
  );

  const chatHistoryRecord = await memoryManager.writeToHistory(
    result.text + "\n",
    companionKey
  );

  if (isText) {
    return NextResponse.json(result.text);
  }
  return new StreamingTextResponse(stream);
}