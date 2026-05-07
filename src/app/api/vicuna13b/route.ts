import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { CallbackManager } from "langchain/callbacks";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

dotenv.config({ path: `.env.local` });

const ALLOWED_MODEL_IDS = ["meta/llama-2-13b-chat", "gpt-3.5-turbo"];
const ALLOWED_COMPANIONS = [
  "Alex.txt",
  "Evelyn.txt",
  "Lucky.txt",
  "Rosie.txt",
  "Sebastian.txt",
];

function sanitizeInput(input: string, maxLength = 8000): string {
  if (!input) return "";
  // Strip null bytes and control characters (except newlines and tabs)
  let sanitized = input.replace(/\0/g, "").replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip dangerous code execution primitives
  sanitized = sanitized.replace(/\b(eval|exec|subprocess|os\.system|child_process|spawn|execSync|spawnSync|execFile)\s*\(/gi, "[REDACTED](");
  // Strip shell command patterns
  sanitized = sanitized.replace(/(`[^`]*`|\$\([^)]*\))/g, "[REDACTED]");
  // Strip invisible Unicode characters
  sanitized = sanitized.replace(/[\u200B-\u200D\uFEFF\u00AD\u2060]/g, "");
  // Truncate to max length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength);
  }
  return sanitized;
}

function sanitizeName(name: string): string {
  // Allow only alphanumeric characters, hyphens, and underscores (prevent path traversal)
  return name.replace(/[^a-zA-Z0-9_-]/g, "");
}

function validateFileContent(content: string): void {
  // Check for hidden/invisible control characters
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content)) {
    throw new Error("Companion file contains invalid control characters.");
  }
  // Check for base64-encoded payloads (long base64 strings)
  if (/[A-Za-z0-9+/]{100,}={0,2}/.test(content)) {
    throw new Error("Companion file contains suspicious base64-encoded content.");
  }
  // Check for shell commands
  if (/\b(eval|exec|subprocess|os\.system|child_process|spawn|execSync|spawnSync|execFile)\s*\(/.test(content)) {
    throw new Error("Companion file contains suspicious shell command patterns.");
  }
  // Check for binary content
  if (/[\x80-\xFF]/.test(content)) {
    throw new Error("Companion file contains binary content.");
  }
  // Check for invisible Unicode characters
  if (/[\u200B-\u200D\uFEFF\u00AD\u2060]/.test(content)) {
    throw new Error("Companion file contains hidden Unicode characters.");
  }
  // Check for leetspeak patterns combined with suspicious keywords
  if (/[3@!1|0]{4,}/.test(content)) {
    throw new Error("Companion file contains suspicious leetspeak patterns.");
  }
}

function sanitizeOutput(output: string): string {
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bsubprocess\b/gi,
    /\bos\.system\s*\(/gi,
    /\bchild_process\b/gi,
    /\bspawn\s*\(/gi,
    /\bexecSync\s*\(/gi,
    /\bspawnSync\s*\(/gi,
    /\bexecFile\s*\(/gi,
    /(`[^`]*`|\$\([^)]*\))/g,
  ];
  for (const pattern of dangerousPatterns) {
    if (pattern.test(output)) {
      throw new Error("LLM output contains dangerous code execution primitives.");
    }
  }
  let sanitized = output;
  for (const pattern of dangerousPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED]");
  }
  return sanitized;
}

export async function POST(request: Request) {
  const { prompt: rawPrompt, isText, userName } = await request.json();

  // Always authenticate server-side via Clerk
  const user = await currentUser();
  const clerkUserId = user?.id;
  const clerkUserName = isText ? (userName || user?.firstName) : user?.firstName;

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

  if (!clerkUserId) {
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

  // Sanitize and validate the name header (prevent path traversal)
  const rawName = request.headers.get("name") || "";
  const name = sanitizeName(rawName);
  const companion_file_name = name + ".txt";

  // Validate companion file name against allow list
  if (!ALLOWED_COMPANIONS.includes(companion_file_name)) {
    return new NextResponse(
      JSON.stringify({ Message: "Companion not permitted." }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Sanitize prompt input
  const prompt = sanitizeInput(rawPrompt || "");

  // Load character "PREAMBLE" from character file.
  const fs = require("fs").promises;
  let data: string;
  try {
    data = await fs.readFile("companions/" + companion_file_name, "utf8");
  } catch (err) {
    return new NextResponse(
      JSON.stringify({ Message: "Companion file not found." }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Validate file content for malicious patterns
  try {
    validateFileContent(data);
  } catch (err: any) {
    return new NextResponse(
      JSON.stringify({ Message: err.message || "Companion file failed validation." }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0]);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "vicuna13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const { stream, handlers } = LangChainStream();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  await memoryManager.writeToHistory(
    "### Human: " + prompt + "\n",
    companionKey
  );

  // Query Pinecone
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);
  recentChatHistory = sanitizeInput(recentChatHistory);

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = sanitizeInput(
      similarDocs.map((doc) => doc.pageContent).join("\n")
    );
  }

  // Validate model ID against allow list
  const selectedModelId = "gpt-3.5-turbo";
  if (!ALLOWED_MODEL_IDS.includes(selectedModelId)) {
    return new NextResponse(
      JSON.stringify({ Message: "Model not permitted." }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Call OpenAI for inference (approved model replacing disallowed vicuna-13b)
  const model = new ChatOpenAI({
    modelName: selectedModelId,
    openAIApiKey: process.env.OPENAI_API_KEY,
    streaming: true,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  const llmPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  // Log the prompt before sending to LLM
  console.log("INFO: Sending prompt to LLM", JSON.stringify({ model: selectedModelId, prompt: llmPrompt }));

  let rawResp: string;
  try {
    rawResp = String(
      await model.call([
        { role: "user", content: llmPrompt } as any,
      ])
    );
  } catch (err) {
    console.error("ERROR: LLM call failed", err);
    return new NextResponse(
      JSON.stringify({ Message: "LLM call failed." }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Log the response received from LLM
  console.log("INFO: Received response from LLM", JSON.stringify({ model: selectedModelId, response: rawResp }));

  // Sanitize and validate LLM output
  let sanitizedResp: string;
  try {
    sanitizedResp = sanitizeOutput(rawResp);
  } catch (err: any) {
    console.error("ERROR: LLM output failed safety check", err.message);
    return new NextResponse(
      JSON.stringify({ Message: "LLM response failed safety validation." }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  const cleaned = sanitizedResp.replaceAll(",", "");
  const chunks = cleaned.split("###");
  const response = chunks[0];

  await memoryManager.writeToHistory("### " + response.trim(), companionKey);
  var Readable = require("stream").Readable;

  let s = new Readable();
  s.push(response);
  s.push(null);
  if (response !== undefined && response.length > 1) {
    await memoryManager.writeToHistory("### " + response.trim(), companionKey);
  }

  return new StreamingTextResponse(s);
}