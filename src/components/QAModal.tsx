"use client";

import {Fragment, useEffect, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

var last_name = "";

const MAX_INPUT_LENGTH = 1000;

function sanitizeInput(value: string): string {
  return value
    .trim()
    .replace(/[<>]/g, "")
    .slice(0, MAX_INPUT_LENGTH);
}

function containsDangerousCode(text: string): boolean {
  const dangerousPatterns = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*["'`]/i,
    /\bsetInterval\s*\(\s*["'`]/i,
    /\bimportScripts\s*\(/i,
    /\bdocument\.write\s*\(/i,
    /\bwindow\s*\[\s*["'`]/i,
    /javascript\s*:/i,
  ];
  return dangerousPatterns.some((pattern) => pattern.test(text));
}

function sanitizeLLMOutput(completion: string): string {
  if (containsDangerousCode(completion)) {
    console.warn("[LLM Output Sanitization] Dangerous content detected in LLM response. Replacing with safe message.");
    return "I'm sorry, but I can't display that response as it contains potentially unsafe content.";
  }
  return completion;
}

export default function QAModal({
  open,
  setOpen,
  example,
}: {
  open: boolean;
  setOpen: any;
  example: any;
}) {
  if (!example) {
    // create a dummy so the completion doesn't croak during init.
    example = new Object();
    example.llm = "";
    example.name = "";
  }

  let {
    completion,
    input,
    isLoading,
    handleInputChange,
    handleSubmit,
    stop,
    setInput,
    setCompletion,
  } = useCompletion({
    api: "/api/" + example.llm,
    headers: { name: example.name },
  });

  let [blocks, setBlocks] = useState<any[] | null>(null)

  useEffect(() => {
    // When the completion changes, parse it to multimodal blocks for display.
    if (completion) {
      console.log("[LLM Interaction Log] Response received:", { companion: example.name, completion });
      const safeCompletion = sanitizeLLMOutput(completion);
      setBlocks(responseToChatBlocks(safeCompletion))
    } else {
      setBlocks(null)
    }
  }, [completion])

  if (!example) {
    console.log("ERROR: no companion selected");
    return null;
  }

  const handleClose = () => {
    setInput("");
    setCompletion("");
    stop();
    setOpen(false);
  };

  const handleCustomSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const sanitized = sanitizeInput(input);

    if (!sanitized) {
      console.warn("[Input Validation] Empty input after sanitization. Submission rejected.");
      return;
    }

    if (sanitized.length > MAX_INPUT_LENGTH) {
      console.warn("[Input Validation] Input exceeds maximum length. Submission rejected.");
      return;
    }

    if (sanitized !== input) {
      setInput(sanitized);
    }

    console.log("[LLM Interaction Log] Request submitted:", { companion: example.name, input: sanitized });

    handleSubmit(e);
  };

  return (
    <Transition.Root show={open} as={Fragment}>
      <Dialog as="div" className="relative z-10" onClose={handleClose}>
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
                  <form onSubmit={handleCustomSubmit}>
                    <input
                      placeholder="How's your day?"
                      className={"w-full flex-auto rounded-md border-0 bg-white/5 px-3.5 py-2 shadow-sm focus:outline-none sm:text-sm sm:leading-6 " + (isLoading && !completion ? "text-gray-600 cursor-not-allowed" : "text-white")}                      
                      value={input}
                      onChange={handleInputChange}
                      disabled={isLoading && !blocks}
                    />
                  </form>
                  <div className="mt-3 sm:mt-5">
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Chat with {example.name}
                      </p>
                    </div>
                    {blocks && (
                      <div className="mt-2">
                        {blocks}
                      </div>
                    )}

                    {isLoading && !blocks && (
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
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}