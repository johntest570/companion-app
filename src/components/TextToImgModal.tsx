import dotenv from "dotenv";

dotenv.config({ path: `.env.local` });

import { Fragment, useState } from "react";
import { Dialog, Transition } from "@headlessui/react";
import Image from "next/image";

const SAFE_URL_PATTERN = /^(https:\/\/|data:image\/[a-zA-Z]+;base64,)[^\s]*$/;
const DANGEROUS_PRIMITIVES = /\b(eval|exec|Function|setTimeout|setInterval)\b/;
const MAX_PROMPT_LENGTH = 500;

function sanitizeAndValidatePrompt(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) {
    console.warn("Prompt is empty after trimming.");
    return null;
  }
  const stripped = trimmed.replace(/[<>"'`\\]/g, "");
  if (!stripped) {
    console.warn("Prompt is empty after stripping dangerous characters.");
    return null;
  }
  if (stripped.length > MAX_PROMPT_LENGTH) {
    console.warn("Prompt exceeds maximum allowed length.");
    return null;
  }
  return stripped;
}

function validateAndSanitizeImageOutput(value: unknown): string | null {
  if (typeof value !== "string" || !value) {
    console.warn("MCP server output is not a non-empty string.");
    return null;
  }
  if (DANGEROUS_PRIMITIVES.test(value)) {
    console.warn("MCP server output contains dynamic code execution primitives. Rejecting.");
    return null;
  }
  if (!SAFE_URL_PATTERN.test(value)) {
    console.warn("MCP server output does not match a safe URL pattern. Rejecting.");
    return null;
  }
  return value;
}

export default function TextToImgModal({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: any;
}) {
  const [imgSrc, setImgSrc] = useState("");
  const [loading, setLoading] = useState(false);
  const [promptValue, setPromptValue] = useState("");

  const onSubmit = async (e: any) => {
    e.preventDefault();

    const sanitizedPrompt = sanitizeAndValidatePrompt(promptValue);
    if (!sanitizedPrompt) {
      console.warn("Invalid or empty prompt. Aborting request.");
      return;
    }

    setLoading(true);

    const requestPayload = {
      prompt: sanitizedPrompt,
    };
    console.log("Request payload to /api/txt2img:", requestPayload);

    const response = await fetch("/api/txt2img", {
      method: "POST",
      body: JSON.stringify(requestPayload),
      headers: {
        "Content-Type": "application/json",
      },
    });
    const data = await response.json();
    console.log("Response data from /api/txt2img:", data);

    const validatedSrc = validateAndSanitizeImageOutput(data[0]);
    if (validatedSrc) {
      setImgSrc(validatedSrc);
    } else {
      console.warn("Image source validation failed. Image will not be displayed.");
    }
    setLoading(false);
  };
  return (
    <Transition.Root show={open} as={Fragment}>
      <Dialog as="div" className="relative z-10" onClose={setOpen}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-950 bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 overflow-y-auto">
          <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-gray-800 px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:p-6 w-full max-w-3xl">
                <div>
                  <input
                    className="w-full flex-auto rounded-md border-0 bg-white/5 px-3.5 py-2 text-white shadow-sm focus:outline-none  sm:text-sm sm:leading-6"
                    placeholder="Describe the image you want"
                    value={promptValue}
                    onChange={(e) => setPromptValue(e.target.value)}
                    // when user click enter key, submit the form
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        onSubmit(e);
                      }
                    }}
                  ></input>
                  <div className="mt-3">
                    <div className="my-2">
                      <p className="text-sm text-gray-500">
                        Powered by{" "}
                        <a
                          className="underline"
                          href="https://replicate.com/stability-ai/stable-diffusion"
                        >
                          stability-ai/stable-diffusion
                        </a>
                      </p>
                    </div>
                  </div>
                </div>
                {imgSrc && !loading && (
                  <Image
                    width={0}
                    height={0}
                    sizes="100vw"
                    src={imgSrc}
                    alt="img"
                    className="w-full h-full object-contain"
                  />
                )}
                {loading && (
                  <p className="flex items-center justify-center mt-4">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        stroke-width="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                  </p>
                )}
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}