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

function sanitizeInput(input: string): string {
  // Strip null bytes
  let sanitized = input.replace(/\0/g, "");
  // Trim whitespace
  sanitized = sanitized.trim();
  // Remove prompt-injection patterns
  sanitized = sanitized
    .split("\n")
    .filter((line) => {
      const trimmed = line.trimStart();
      return (
        !trimmed.startsWith("SYSTEM:") &&
        !trimmed.startsWith("INST:") &&
        !trimmed.startsWith("###") &&
        !trimmed.startsWith("[INST]")
      );
    })
    .join("\n");
  return sanitized;
}

function sanitizeFileContent(content: string): string {
  const MAX_LENGTH = 50000;

  // Reject binary/non-UTF-8 content (check for replacement character)
  if (content.includes("\uFFFD")) {
    throw new Error("Invalid file content: binary or non-UTF-8 data detected");
  }

  // Strip invisible/zero-width Unicode characters
  content = content.replace(
    /[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060]/g,
    ""
  );

  // Strip or reject base64-encoded blobs (long base64 strings)
  content = content.replace(/[A-Za-z0-9+/]{100,}={0,2}/g, "");

  // Detect and reject shell command patterns
  const shellPatterns = /(\$\(|\`|;\s*rm\s|;\s*curl\s|;\s*wget\s|&&\s*rm\s)/i;
  if (shellPatterns.test(content)) {
    throw new Error("Invalid file content: shell command patterns detected");
  }

  // Detect and reject common prompt-injection trigger phrases
  const injectionPatterns =
    /ignore previous instructions|system:|you are now|forget your instructions|disregard your/i;
  if (injectionPatterns.test(content)) {
    throw new Error(
      "Invalid file content: prompt injection patterns detected"
    );
  }

  // Enforce length cap
  if (content.length > MAX_LENGTH) {
    content = content.substring(0, MAX_LENGTH);
  }

  // Also apply sanitizeInput patterns
  content = sanitizeInput(content);

  return content;
}

function sanitizeLLMOutput(output: string): string {
  const dangerousPatterns =
    /\beval\s*\(|\bexec\s*\(|\bsubprocess\b|\bos\.system\s*\(|\bspawn\s*\(|\bFunction\s*\(|\bnew\s+Function\b/i;
  if (dangerousPatterns.test(output)) {
    console.warn(
      "WARNING: LLM output contained dangerous code execution primitives, sanitizing."
    );
    return "[Response blocked due to unsafe content]";
  }
  return output;
}

export async function POST(request: Request) {
  const { prompt, isText, userId, userName } = await request.json();
  let clerkUserId;
  let user;
  let clerkUserName;

  // Validate and sanitize userId
  if (isText) {
    if (
      !userId ||
      typeof userId !== "string" ||
      userId.trim().length === 0 ||
      !/^[a-zA-Z0-9_\-]+$/.test(userId.trim())
    ) {
      return new NextResponse(
        JSON.stringify({ Message: "Invalid userId" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    if (
      !userName ||
      typeof userName !== "string" ||
      userName.trim().length === 0 ||
      !/^[a-zA-Z0-9_\- ]+$/.test(userName.trim())
    ) {
      return new NextResponse(
        JSON.stringify({ Message: "Invalid userName" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
  }

  // Sanitize prompt
  if (!prompt || typeof prompt !== "string") {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const sanitizedPrompt = sanitizeInput(prompt.substring(0, 4096));

  const identifier = request.url + "-" + (userId || "anonymous");
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
  const rawName = request.headers.get("name");

  // Validate name to only allow alphanumeric characters and hyphens/underscores
  if (!rawName || !/^[a-zA-Z0-9_\-]+$/.test(rawName)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const name = rawName;
  const companion_file_name = name + ".txt";

  if (isText) {
    clerkUserId = userId.trim();
    clerkUserName = userName.trim();
  } else {
    user = await currentUser();
    clerkUserId = user?.id;
    clerkUserName = user?.firstName;
  }

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
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
  const data = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  let preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  let seedchat = seedsplit[0];

  // Sanitize file content to prevent malicious prompt injection
  try {
    preamble = sanitizeFileContent(preamble);
    seedchat = sanitizeFileContent(seedchat);
  } catch (e: any) {
    console.error("File content sanitization failed:", e.message);
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion file content" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

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
  await memoryManager.writeToHistory(
    "User: " + sanitizedPrompt + "\n",
    companionKey
  );

  // Query chat history only (no Pinecone vector search to limit external credentials)
  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  // Sanitize recentChatHistory before embedding in prompt
  const sanitizedRecentChatHistory = sanitizeInput(recentChatHistory);

  // No vector search - use empty relevantHistory to reduce external system credentials
  const relevantHistory = "";

  const { stream, handlers } = LangChainStream();
  // Call OpenAI GPT-3.5-turbo for inference (approved model)
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo-instruct",
    maxTokens: 2048,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const llmPrompt = `
       ONLY generate NO more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${relevantHistory}


       ${sanitizedRecentChatHistory}\n${name}:`;

  console.log("INFO: Sending prompt to LLM:", llmPrompt);

  let resp = String(
    await model
      .call(llmPrompt)
      .catch(console.error)
  );

  console.log("INFO: Received response from LLM:", resp);

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  const cleaned = resp.replaceAll(",", "");
  const chunks = cleaned.split("\n");
  let response = chunks[0];
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

  // Sanitize LLM output for dangerous code execution primitives
  response = sanitizeLLMOutput(response);

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