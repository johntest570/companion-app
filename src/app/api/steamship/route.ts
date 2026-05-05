import dotenv from "dotenv";
import clerk from "@clerk/clerk-sdk-node";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
import {Md5} from 'ts-md5'
import ConfigManager from "@/app/utils/config";

dotenv.config({ path: `.env.local` });

const MAX_PROMPT_LENGTH = 4096;
const SAFE_ID_PATTERN = /^[a-zA-Z0-9_\-\.@]+$/;
const SAFE_NAME_PATTERN = /^[a-zA-Z0-9_ \-\.]+$/;
const DANGEROUS_PRIMITIVES = /\beval\b|\bexec\b|\bFunction\b|\bnew\s+Function\b|\bsetTimeout\b|\bsetInterval\b/;

function returnError(code: number, message: string) {
  return new NextResponse(
      JSON.stringify({ Message: message }),
      {
        status: code,
        headers: {
          "Content-Type": "application/json",
        },
      }
  );
}

function sanitizePrompt(raw: string): string {
  // Strip null bytes and control characters (except common whitespace)
  return raw.replace(/\0/g, "").replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "").trim();
}

function containsDangerousPrimitives(obj: unknown): boolean {
  if (typeof obj === "string") {
    return DANGEROUS_PRIMITIVES.test(obj);
  }
  if (Array.isArray(obj)) {
    return obj.some((item) => containsDangerousPrimitives(item));
  }
  if (obj !== null && typeof obj === "object") {
    return Object.values(obj as Record<string, unknown>).some((val) => containsDangerousPrimitives(val));
  }
  return false;
}

const ALLOWED_BLOCK_FIELDS = new Set(["id", "text", "mimeType", "url"]);

function sanitizeResponseBlocks(data: unknown): unknown {
  if (Array.isArray(data)) {
    return data.map((item) => sanitizeResponseBlocks(item));
  }
  if (data !== null && typeof data === "object") {
    const sanitized: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
      if (ALLOWED_BLOCK_FIELDS.has(key)) {
        sanitized[key] = value;
      }
    }
    return sanitized;
  }
  return data;
}

export async function POST(req: Request) {
  let clerkUserId;
  let user;
  let clerkUserName;
  const { prompt: rawPrompt, isText, userId, userName } = await req.json();
  const companionName = req.headers.get("name");

  // Validate and sanitize prompt
  if (!rawPrompt || typeof rawPrompt !== "string") {
    return returnError(400, "Prompt is required and must be a non-empty string.");
  }
  const sanitizedPrompt = sanitizePrompt(rawPrompt);
  if (!sanitizedPrompt || sanitizedPrompt.length === 0) {
    return returnError(400, "Prompt must be a non-empty string after sanitization.");
  }
  if (sanitizedPrompt.length > MAX_PROMPT_LENGTH) {
    return returnError(400, `Prompt exceeds the maximum allowed length of ${MAX_PROMPT_LENGTH} characters.`);
  }

  // Validate userId
  if (userId !== undefined && userId !== null) {
    if (typeof userId !== "string" || !SAFE_ID_PATTERN.test(userId)) {
      return returnError(400, "Invalid userId format.");
    }
  }

  // Validate userName
  if (userName !== undefined && userName !== null) {
    if (typeof userName !== "string" || !SAFE_NAME_PATTERN.test(userName)) {
      return returnError(400, "Invalid userName format.");
    }
  }

  const prompt = sanitizedPrompt;

  if (!companionName) {
    console.log("ERROR: no companion name");
    return returnError(429, `Hi, please add a 'name' field in your headers specifying the Companion Name.`)
  }

  // Load the companion config
  const configManager = ConfigManager.getInstance();
  const companionConfig = configManager.getConfig("name", companionName);
  if (!companionConfig) {
    return returnError(404, `Hi, we were unable to find the configuration for a companion named ${companionName}.`)
  }

  // Make sure we're not rate limited
  const identifier = req.url + "-" + (userId || "anonymous");
  const { success } = await rateLimit(identifier);
  if (!success) {
    console.log("INFO: rate limit exceeded");
    return returnError(429, `Hi, the companions can't talk this fast.`)
  }

  if (!process.env.STEAMSHIP_API_KEY) {
    return returnError(500, `Please set the STEAMSHIP_API_KEY env variable and make sure ${companionName} is connected to an Agent instance that you own.`)
  }

  console.log(`Companion Name: ${companionName}`)
  console.log(`Prompt: ${prompt}`);

  if (isText) {
    clerkUserId = userId;
    clerkUserName = userName;
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

  // Policy: Block requests to Steamship-hosted GPT-backed agent endpoint — not on approved LLM list
  return returnError(403, "The Steamship-hosted GPT-backed agent is not on the organization's approved LLM list. Request blocked by policy.");

  // Create a chat session id for the user
  const chatSessionId = Md5.hashStr(userId || "anonymous");

  // Make sure we have a generate endpoint.
  // TODO: Create a new instance of the agent per user if this proves advantageous.
  const agentUrl = companionConfig.generateEndpoint
  if (!agentUrl) {
    return returnError(500, `Please add a Steamship 'generateEndpoint' to your ${companionName} configuration in companions.json.`)
  }

  const requestPayload = {
    question: prompt,
    chat_session_id: chatSessionId
  };

  // Log MCP server request
  console.log(`MCP Server Request URL: ${agentUrl}`);
  console.log(`MCP Server Request Payload: ${JSON.stringify(requestPayload)}`);

  // Log LLM interaction request
  console.log(`LLM Interaction Request - agentUrl: ${agentUrl}, payload: ${JSON.stringify(requestPayload)}`);

  // Invoke the generation. Tool invocation, chat history management, backstory injection, etc is all done within this endpoint.
  // To build, deploy, and host your own multi-tenant agent see: https://www.steamship.com/learn/agent-guidebook
  const response = await fetch(agentUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.STEAMSHIP_API_KEY}`
    },
    body: JSON.stringify(requestPayload)
  });

  if (response.ok) {
    const responseText = await response.text();

    // Log MCP server response
    console.log(`MCP Server Response Status: ${response.status}`);
    console.log(`MCP Server Response Body: ${responseText}`);

    // Log LLM interaction response
    console.log(`LLM Interaction Response - status: ${response.status}, body: ${responseText}`);

    let responseBlocks: unknown;
    try {
      responseBlocks = JSON.parse(responseText);
    } catch (e) {
      return returnError(500, "Failed to parse response from agent endpoint.");
    }

    // Validate structure
    if (typeof responseBlocks !== "object" || responseBlocks === null) {
      return returnError(500, "Unexpected response structure from agent endpoint.");
    }

    // Check for dangerous primitives in LLM output
    if (containsDangerousPrimitives(responseBlocks)) {
      console.log("ERROR: LLM response contains dangerous code execution primitives.");
      return returnError(500, "LLM response contained potentially dangerous content and was blocked.");
    }

    // Sanitize output: allowlist only known safe fields
    const sanitizedBlocks = sanitizeResponseBlocks(responseBlocks);

    return NextResponse.json(sanitizedBlocks);
  } else {
    const errorText = await response.text();

    // Log MCP server error response
    console.log(`MCP Server Response Status: ${response.status}`);
    console.log(`MCP Server Response Body: ${errorText}`);

    // Log LLM interaction error response
    console.log(`LLM Interaction Response - status: ${response.status}, body: ${errorText}`);

    return returnError(500, errorText);
  }
}