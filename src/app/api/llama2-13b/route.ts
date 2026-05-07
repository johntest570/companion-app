import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { OpenAI } from "langchain/llms/openai";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

dotenv.config({ path: `.env.local` });

const POLICY_VERSION = "1.0.0";
const ALLOWED_COMPANIONS = ["Alex", "Evelyn", "Lucky", "Rosie", "Sebastian"];
const ALLOWED_MODELS = ["gpt-3.5-turbo"];
const SELECTED_MODEL = "gpt-3.5-turbo";

const DANGEROUS_PATTERNS = [
  /\beval\s*\(/i,
  /\bexec\s*\(/i,
  /\bsubprocess\b/i,
  /\bos\.system\s*\(/i,
  /\bspawn\s*\(/i,
  /\bshell\s*=\s*True/i,
  /\bpopen\s*\(/i,
  /\b__import__\s*\(/i,
  /\bimportlib\b/i,
  /\bFunction\s*\(/i,
  /\bnew\s+Function\b/i,
  /\bsetTimeout\s*\(\s*["'`]/i,
  /\bsetInterval\s*\(\s*["'`]/i,
];

const INJECTION_PATTERNS = [
  /[;\|&`\$\(\)\{\}><]/,
  /base64/i,
  /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/,
];

const MAX_INPUT_LENGTH = 4000;

function sanitizeInput(input: string): string {
  if (!input) return "";
  // Remove null bytes and control characters
  let sanitized = input.replace(/\x00/g, "").replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
  // Truncate excessively long strings
  if (sanitized.length > MAX_INPUT_LENGTH) {
    sanitized = sanitized.substring(0, MAX_INPUT_LENGTH);
  }
  return sanitized;
}

function containsDangerousContent(text: string): boolean {
  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(text)) {
      return true;
    }
  }
  return false;
}

function containsInjectionAttempt(text: string): boolean {
  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(text)) {
      return true;
    }
  }
  return false;
}

function sanitizeCompanionName(name: string): string | null {
  if (!name) return null;
  // Allow only alphanumeric characters, hyphens, and underscores
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return null;
  }
  return name;
}

function logAudit(entry: Record<string, unknown>): void {
  console.log(JSON.stringify({ audit: true, timestamp: new Date().toISOString(), policyVersion: POLICY_VERSION, ...entry }));
}

export async function POST(request: Request) {
  const { prompt: rawPrompt, isText } = await request.json();

  // Always use server-side Clerk authentication regardless of isText
  const user = await currentUser();
  const clerkUserId = user?.id;
  const clerkUserName = user?.firstName;

  const identifier = request.url + "-" + (clerkUserId || "anonymous");
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

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
    logAudit({ event: "auth_failure", userId: clerkUserId });
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

  // Sanitize and validate the companion name header to prevent path traversal
  const rawName = request.headers.get("name");
  const name = rawName ? sanitizeCompanionName(rawName) : null;

  if (!name) {
    logAudit({ event: "invalid_companion_name", rawName, userId: clerkUserId });
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Validate companion name against allow list
  if (!ALLOWED_COMPANIONS.includes(name)) {
    logAudit({ event: "companion_not_allowed", companion: name, userId: clerkUserId, outcome: "denied" });
    return new NextResponse(
      JSON.stringify({ Message: "Companion not permitted by policy." }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Validate model against allow list
  if (!ALLOWED_MODELS.includes(SELECTED_MODEL)) {
    logAudit({ event: "model_not_allowed", model: SELECTED_MODEL, userId: clerkUserId, outcome: "denied" });
    return new NextResponse(
      JSON.stringify({ Message: "Model not permitted by policy." }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const companion_file_name = name + ".txt";

  // Sanitize prompt
  const prompt = sanitizeInput(rawPrompt || "");

  if (containsInjectionAttempt(prompt)) {
    logAudit({ event: "injection_attempt_in_prompt", userId: clerkUserId });
    return new NextResponse(
      JSON.stringify({ Message: "Input contains disallowed content." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const fs = require("fs").promises;
  const data = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0]);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "llama2-13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  await memoryManager.writeToHistory("User: " + prompt + "\n", companionKey);

  // Query Pinecone
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);
  const sanitizedRecentChatHistory = sanitizeInput(recentChatHistory);

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = sanitizeInput(similarDocs.map((doc) => doc.pageContent).join("\n"));
  }

  const { stream, handlers } = LangChainStream();

  // Call OpenAI for inference (approved model)
  const model = new OpenAI({
    modelName: SELECTED_MODEL,
    maxTokens: 2048,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const fullPrompt = `
       ONLY generate NO more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${relevantHistory}


       ${sanitizedRecentChatHistory}\n${name}:`;

  // Log the prompt before sending to the model
  console.log(JSON.stringify({
    event: "llm_request",
    timestamp: new Date().toISOString(),
    userId: clerkUserId,
    model: SELECTED_MODEL,
    companion: name,
    prompt: fullPrompt,
  }));

  logAudit({
    event: "tool_invocation",
    actor: clerkUserId,
    model: SELECTED_MODEL,
    companion: name,
    outcome: "attempt",
  });

  let resp: string;
  try {
    resp = String(await model.call(fullPrompt));
  } catch (err) {
    logAudit({
      event: "tool_invocation",
      actor: clerkUserId,
      model: SELECTED_MODEL,
      companion: name,
      outcome: "failure",
      error: String(err),
    });
    console.error("LLM call failed:", err);
    return new NextResponse(
      JSON.stringify({ Message: "Model inference failed." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Log the response received from the model
  console.log(JSON.stringify({
    event: "llm_response",
    timestamp: new Date().toISOString(),
    userId: clerkUserId,
    model: SELECTED_MODEL,
    companion: name,
    response: resp,
  }));

  logAudit({
    event: "tool_invocation",
    actor: clerkUserId,
    model: SELECTED_MODEL,
    companion: name,
    outcome: "success",
  });

  // Validate and sanitize LLM output for dangerous content
  if (containsDangerousContent(resp)) {
    console.error("LLM output contains dangerous content, blocking response.");
    logAudit({ event: "dangerous_output_blocked", userId: clerkUserId, model: SELECTED_MODEL, companion: name });
    return new NextResponse(
      JSON.stringify({ Message: "Response blocked due to policy violation." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Right now just using super shoddy string manip logic to get at
  // the dialog.
  const cleaned = resp.replaceAll(",", "");
  const chunks = cleaned.split("\n");
  const response = chunks[0];

  await memoryManager.writeToHistory("" + response.trim(), companionKey);
  var Readable = require("stream").Readable;

  let s = new Readable();
  s.push(response);
  s.push(null);
  if (response !== undefined && response.length > 1) {
    memoryManager.writeToHistory("" + response.trim(), companionKey);
  }

  return new StreamingTextResponse(s);
}