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

const MAX_PROMPT_LENGTH = 4096;
const DANGEROUS_PATTERNS = /\b(eval|exec|subprocess|Function\s*\(|new\s+Function|setTimeout\s*\(|setInterval\s*\(|execSync|spawnSync|child_process)\b/i;

function sanitizeInput(input: string, maxLength: number = MAX_PROMPT_LENGTH): string {
  if (typeof input !== "string") return "";
  return input.trim().slice(0, maxLength).replace(/[<>]/g, "");
}

function validateName(name: string | null): boolean {
  if (!name) return false;
  return /^[a-zA-Z0-9_-]+$/.test(name);
}

function validateUserId(userId: string): boolean {
  if (!userId || typeof userId !== "string") return false;
  return /^[a-zA-Z0-9_\-\.@]+$/.test(userId.trim()) && userId.trim().length <= 128;
}

function validateUserName(userName: string): boolean {
  if (!userName || typeof userName !== "string") return false;
  return /^[a-zA-Z0-9_ \-\.]+$/.test(userName.trim()) && userName.trim().length <= 64;
}

function containsDangerousPatterns(output: string): boolean {
  return DANGEROUS_PATTERNS.test(output);
}

export async function POST(request: Request) {
  const { prompt, isText, userId, userName } = await request.json();
  let clerkUserId;
  let user;
  let clerkUserName;

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
  const name = request.headers.get("name");

  if (!validateName(name)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  const companion_file_name = name + ".txt";

  if (isText) {
    if (!validateUserId(userId)) {
      return new NextResponse(
        JSON.stringify({ Message: "Invalid userId format." }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    if (!validateUserName(userName)) {
      return new NextResponse(
        JSON.stringify({ Message: "Invalid userName format." }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    clerkUserId = userId;
    clerkUserName = userName;
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
  const preamble = sanitizeInput(presplit[0], 8192);
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

  const sanitizedPrompt = sanitizeInput(prompt, MAX_PROMPT_LENGTH);

  await memoryManager.writeToHistory(
    "### Human: " + sanitizedPrompt + "\n",
    companionKey
  );

  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);

  // Pinecone vector search removed to reduce external system credentials to three
  let relevantHistory = "";

  const sanitizedRelevantHistory = sanitizeInput(relevantHistory, 4096);

  // Call OpenAI for inference
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const llmPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${sanitizedRelevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  console.log("INFO: LLM input prompt:", llmPrompt);

  let resp = String(
    await model
      .call(llmPrompt)
      .catch(console.error)
  );

  console.log("INFO: LLM output response:", resp);

  if (containsDangerousPatterns(resp)) {
    console.log("INFO: Dangerous pattern detected in LLM output. Rejecting response.");
    return new NextResponse(
      JSON.stringify({ Message: "Response rejected due to policy violation." }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  const cleaned = resp.replaceAll(",", "");
  const chunks = cleaned.split("###");
  const response = chunks[0];
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

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