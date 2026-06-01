import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { Replicate, ReplicateInput } from "langchain/llms/replicate";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

dotenv.config({ path: `.env.local` });

export async function POST(request: Request) {
  const sanitizeInput = (input: string) => input?.replace(/[${}`\\]/g, '\\$&') || '';
  const { prompt, isText, userId, userName } = await request.json();
  if (!prompt) {
    return new NextResponse(
      JSON.stringify({ Message: "Missing required 'prompt' parameter" }),
      {
        status: 400,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }
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
  const name = sanitizeInput(request.headers.get("name"));
  const companion_file_name = name + ".txt";

  if (isText) {
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
  if (presplit.length < 2) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion file format - missing preamble" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  const preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  if (seedsplit.length < 2) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion file format - missing seed chat" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
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

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs.map((doc) => doc.pageContent).join("\n");
  }
  const { stream, handlers } = LangChainStream();
  // Call Replicate for inference
  const model = new Replicate({
    model:
      "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input: {
      max_length: 2048,
    },
    apiKey: process.env.REPLICATE_API_TOKEN,
    callbackManager: CallbackManager.fromHandlers({
  ...handlers,
  handleLLMEnd: (output) => {
    console.log(JSON.stringify(output));
  },
}),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  let resp = String(
    await model
      .call(
        `
       ONLY generate NO more than three sentences as ${sanitizeInput(name)}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${sanitizeInput(name)}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${sanitizeInput(name)}'s past and the conversation you are in.
       ${relevantHistory}


       ${sanitizeInput(recentChatHistory)}\n${sanitizeInput(name)}:`
      )
      .catch(console.error)
  );

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  const cleaned = resp.replace(/<[^>]*>/g, '').replaceAll(",", "");
  const chunks = cleaned.split("\n");
  let response = chunks[0];
  const forbiddenPatterns = [/eval\(/, /new Function\(/, /setTimeout\(/, /setInterval\(/, /exec\(/];
  const isUnsafe = forbiddenPatterns.some(pattern => pattern.test(response));
  if (isUnsafe) {
    response = "Apologies, I can't assist with that.";
  }
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

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
