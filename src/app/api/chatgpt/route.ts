import { ChatAnthropic } from "langchain/chat_models/anthropic";
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

dotenv.config({ path: `.env.local` });

function sanitizeInput(input: string): string {
  // Remove control characters
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Remove prompt injection patterns
  sanitized = sanitized.replace(
    /\b(ignore previous instructions|disregard|override|system prompt|you are now|act as|pretend you are)\b/gi,
    ""
  );
  // Normalize excessive whitespace
  sanitized = sanitized.replace(/\s{3,}/g, "  ").trim();
  return sanitized;
}

function sanitizeCompanionContent(content: string): string {
  // Check for zero-width / invisible characters
  if (/[\u200B-\u200D\uFEFF\u00AD]/.test(content)) {
    throw new Error("Companion file contains hidden/invisible characters.");
  }
  // Check for base64-encoded payloads (long base64 strings)
  if (/[A-Za-z0-9+/]{100,}={0,2}/.test(content)) {
    throw new Error("Companion file contains potential base64-encoded payload.");
  }
  // Check for suspicious instruction keywords
  if (
    /\b(ignore previous instructions|disregard all|override system|you are now|act as if|pretend you are|system prompt)\b/i.test(
      content
    )
  ) {
    throw new Error("Companion file contains suspicious instruction keywords.");
  }
  // Check for shell command sequences
  if (/(\$\(|`[^`]*`|;\s*(rm|curl|wget|bash|sh)\s)/.test(content)) {
    throw new Error("Companion file contains potential shell command sequences.");
  }
  // Check for binary content
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F]/.test(content)) {
    throw new Error("Companion file contains binary/control characters.");
  }
  return content;
}

function scrubPII(text: string): string {
  // Remove email addresses
  let scrubbed = text.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g, "[EMAIL]");
  // Remove IP addresses
  scrubbed = scrubbed.replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[IP]");
  // Remove SSNs
  scrubbed = scrubbed.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[SSN]");
  // Remove phone numbers
  scrubbed = scrubbed.replace(/\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b/g, "[PHONE]");
  // Remove credit card numbers
  scrubbed = scrubbed.replace(/\b(?:\d[ -]?){13,16}\b/g, "[CC]");
  // Remove medical record patterns (MRN: followed by digits)
  scrubbed = scrubbed.replace(/\b(MRN|mrn|medical record(?: number)?)[:\s#]*\d+\b/gi, "[MRN]");
  return scrubbed;
}

function sanitizeLLMOutput(output: string): string {
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /new\s+Function\s*\(/gi,
    /setTimeout\s*\(\s*["'`]/gi,
    /setInterval\s*\(\s*["'`]/gi,
    /\bimport\s*\(/gi,
    /require\s*\(\s*["'`]/gi,
  ];
  for (const pattern of dangerousPatterns) {
    if (pattern.test(output)) {
      return "I'm sorry, I cannot provide that response.";
    }
  }
  return output;
}

export async function POST(req: Request) {
  let clerkUserId;
  let user;
  let clerkUserName;
  const { prompt, isText, userId, userName } = await req.json();

  // Validate and sanitize inputs from request body
  const sanitizedPrompt = sanitizeInput(
    typeof prompt === "string" ? prompt : ""
  );
  const sanitizedUserId = typeof userId === "string" ? userId.trim().replace(/[^a-zA-Z0-9_\-]/g, "") : "";
  const sanitizedUserName = typeof userName === "string" ? userName.trim().replace(/[<>"'&]/g, "") : "";

  // Validate prompt
  if (!sanitizedPrompt || sanitizedPrompt.length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Prompt is required." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  if (sanitizedPrompt.length > 4000) {
    return new NextResponse(
      JSON.stringify({ Message: "Prompt exceeds maximum allowed length." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const identifier = req.url + "-" + (sanitizedUserId || "anonymous");
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

  // XXX Companion name passed here. Can use as a key to get backstory, chat history etc.
  const rawName = req.headers.get("name");

  // Validate name header: only allow alphanumeric, hyphens, underscores
  if (!rawName || !/^[a-zA-Z0-9_\-]+$/.test(rawName)) {
    console.log("Invalid or missing companion name");
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const name = rawName;
  const companionFileName = name + ".txt";

  console.log("prompt: ", sanitizedPrompt);
  if (isText) {
    clerkUserId = sanitizedUserId;
    clerkUserName = sanitizedUserName;
  } else {
    user = await currentUser();
    clerkUserId = user?.id;
    clerkUserName = user?.firstName;
  }

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

  // Load character "PREAMBLE" from character file. These are the core personality
  // characteristics that are used in every prompt. Additional background is
  // only included if it matches a similarity comparioson with the current
  // discussion. The PREAMBLE should include a seed conversation whose format will
  // vary by the model using it.
  const fs = require("fs").promises;
  const data = await fs.readFile("companions/" + companionFileName, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  let preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  let seedchat = seedsplit[0];

  // Sanitize companion file content to prevent malicious prompt injection
  try {
    preamble = sanitizeCompanionContent(preamble);
    seedchat = sanitizeCompanionContent(seedchat);
  } catch (err: any) {
    console.error("Companion file sanitization failed:", err.message);
    return new NextResponse(
      JSON.stringify({ Message: "Companion file contains invalid content." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Further sanitize preamble and seedchat for injection patterns
  preamble = sanitizeInput(preamble);
  seedchat = sanitizeInput(seedchat);

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

  // Scrub PII from prompt before writing to history
  const piiScrubbedPrompt = scrubPII(sanitizedPrompt);
  await memoryManager.writeToHistory("Human: " + piiScrubbedPrompt + "\n", companionKey);
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  // Scrub PII from recentChatHistory before sending to LLM
  recentChatHistory = scrubPII(recentChatHistory);

  // Remove Pinecone vector search (fourth external system) — use empty relevantHistory
  const relevantHistory = "";

  const { stream, handlers } = LangChainStream();

  const model = new ChatAnthropic({
    streaming: true,
    modelName: "claude-2",
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  model.verbose = true;

  const replyWithTwilioLimit = isText
    ? "You reply within 1000 characters."
    : "";

  // Use generic placeholder instead of real user name to avoid sending PII to LLM
  const chainPrompt = PromptTemplate.fromTemplate(`
    You are ${name} and are currently talking to a user.

    ${preamble}

  You reply with answers that range from one sentence to one paragraph and with some details. ${replyWithTwilioLimit}

  Below are relevant details about ${name}'s past
  ${relevantHistory}
  
  Below is a relevant conversation history

  ${recentChatHistory}`);

  const chain = new LLMChain({
    llm: model,
    prompt: chainPrompt,
  });

  // Log the full constructed prompt before sending to LLM
  const formattedPrompt = await chainPrompt.format({
    relevantHistory,
    recentChatHistory,
  });
  console.log("INFO: Full prompt sent to LLM:", formattedPrompt);

  const result = await chain
    .call({
      relevantHistory,
      recentChatHistory: recentChatHistory,
    })
    .catch(console.error);

  console.log("result", result);

  // Sanitize LLM output before writing to history or returning
  const sanitizedOutput = sanitizeLLMOutput(result!.text);

  const chatHistoryRecord = await memoryManager.writeToHistory(
    sanitizedOutput + "\n",
    companionKey
  );
  console.log("chatHistoryRecord", chatHistoryRecord);
  if (isText) {
    return NextResponse.json(sanitizedOutput);
  }
  return new StreamingTextResponse(stream);
}