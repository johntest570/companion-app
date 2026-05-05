import { NextResponse } from "next/server";
import twilio from "twilio";
import clerk from "@clerk/clerk-sdk-node";
import dotenv from "dotenv";
import ConfigManager from "@/app/utils/config";
import { rateLimit } from "@/app/utils/rateLimit";
import crypto from "crypto";

dotenv.config({ path: `.env.local` });
const twilioAuthToken = process.env.TWILIO_AUTH_TOKEN;
const accountSid = process.env.TWILIO_ACCOUNT_SID;

const APPROVED_MODEL = "llama2-13b";
const APPROVED_MODELS = ["llama2-13b"];

const DANGEROUS_PATTERNS = [
  /\beval\s*\(/gi,
  /\bexec\s*\(/gi,
  /\bsubprocess\b/gi,
  /\bnew\s+Function\s*\(/gi,
  /\bsetTimeout\s*\(\s*["'`]/gi,
  /\bsetInterval\s*\(\s*["'`]/gi,
  /\bFunction\s*\(/gi,
];

function sanitizeLLMOutput(text: string): string {
  let sanitized = text;
  for (const pattern of DANGEROUS_PATTERNS) {
    sanitized = sanitized.replace(pattern, "[REMOVED]");
  }
  return sanitized;
}

function encryptPII(value: string, key: Buffer): { encrypted: string; iv: string; tag: string } {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  let encrypted = cipher.update(value, "utf8", "hex");
  encrypted += cipher.final("hex");
  const tag = cipher.getAuthTag();
  return {
    encrypted,
    iv: iv.toString("hex"),
    tag: tag.toString("hex"),
  };
}

const E164_REGEX = /^\+[1-9]\d{1,14}$/;
const MAX_PROMPT_LENGTH = 1000;

export async function POST(request: Request) {
  let queryMap: any = {};
  const twilioClient = twilio(accountSid, twilioAuthToken);
  const data = decodeURIComponent(await request.text());
  data.split("&").forEach((item) => {
    queryMap[item.split("=")[0]] = item.split("=")[1];
  });

  // Sanitize and validate prompt (Instruction 2, 6)
  let prompt: string = queryMap["Body"] || "";
  prompt = prompt.trim();
  // Strip null bytes and control characters
  prompt = prompt.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  if (!prompt || prompt.length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  if (prompt.length > MAX_PROMPT_LENGTH) {
    prompt = prompt.substring(0, MAX_PROMPT_LENGTH);
  }

  const serverUrl = process.env.INTERNAL_API_BASE_URL || request.url.split("/api/")[0].replace(/^http:/, "https:");
  const phoneNumber = queryMap["From"];
  const companionPhoneNumber = queryMap["To"];

  // Validate phoneNumber (Instruction 2)
  if (!phoneNumber || !E164_REGEX.test(phoneNumber)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing 'From' phone number." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Validate companionPhoneNumber (Instruction 2)
  if (!companionPhoneNumber || !E164_REGEX.test(companionPhoneNumber)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing 'To' phone number." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const identifier = request.url + "-" + (phoneNumber || "anonymous");
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

  // check if the user has verified phone #
  const users = await clerk.users.getUserList({ phoneNumber });

  if (!users || users.length == 0) {
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

  const configManager = ConfigManager.getInstance();
  const companionConfig = configManager.getConfig(
    "phone",
    companionPhoneNumber
  );
  console.log("companionConfig: ", companionConfig);
  if (!companionConfig || companionConfig.length == 0) {
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

  const companionName = companionConfig.name;

  // Use only the approved model, ignoring companionConfig.llm (Instructions 1, 3)
  const companionModel = APPROVED_MODEL;

  // Validate model against allowlist (Instruction 3)
  if (!APPROVED_MODELS.includes(companionModel)) {
    return new NextResponse(
      JSON.stringify({ Message: "Model not approved." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Log outgoing LLM request (Instruction 5)
  console.log("INFO: Sending request to LLM", {
    model: companionModel,
    companionName,
    prompt,
    isText: true,
    timestamp: new Date().toISOString(),
  });

  // Inter-agent authentication (Instruction 9)
  const interAgentApiKey = process.env.INTER_AGENT_API_KEY || "";

  // Build request body without PII (Instruction 8)
  const requestBody = JSON.stringify({
    prompt,
    isText: true,
  });

  const response = await fetch(`${serverUrl}/api/${companionModel}`, {
    body: requestBody,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      name: companionName,
      Authorization: `Bearer ${interAgentApiKey}`,
    },
  });

  const rawResponseText = await response.text();

  // Log LLM response (Instruction 5)
  console.log("INFO: Received response from LLM", {
    model: companionModel,
    responseText: rawResponseText,
    timestamp: new Date().toISOString(),
  });

  // Sanitize LLM output (Instruction 4)
  const responseText = sanitizeLLMOutput(rawResponseText);

  const to = queryMap["From"];
  const from = queryMap["To"];
  console.log("responseText: ", responseText);
  await twilioClient.messages
    .create({
      body: responseText,
      from,
      to,
    })
    .catch((err) => {
      console.log("WARNING: failed to send SMS.", err);
    });

  return NextResponse.json({ message: "Hello from the API!" });
}