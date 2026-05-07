/*
 * Represents a unit of multimodal chat: text, video, audio, or image.
 *
 * For streaming responses, just update the `text` argument.
 */

/*
 * Sanitizes a string value extracted from LLM output by detecting and stripping
 * dangerous dynamic code execution primitives such as eval, exec, Function constructor, etc.
 */
function sanitizeLLMString(value: string): string {
    // Patterns for dangerous dynamic code execution primitives
    const dangerousPatterns = [
        /\beval\s*\(/gi,
        /\bexec\s*\(/gi,
        /\bnew\s+Function\s*\(/gi,
        /\bFunction\s*\(/gi,
        /\bsetTimeout\s*\(\s*["'`]/gi,
        /\bsetInterval\s*\(\s*["'`]/gi,
        /\bsetImmediate\s*\(\s*["'`]/gi,
        /\bexecScript\s*\(/gi,
        /\bimportScripts\s*\(/gi,
        /javascript\s*:/gi,
        /\bdocument\.write\s*\(/gi,
        /\bwindow\.location\s*=/gi,
        /\blocation\.href\s*=/gi,
    ];

    let sanitized = value;
    for (const pattern of dangerousPatterns) {
        sanitized = sanitized.replace(pattern, '[REMOVED]');
    }
    return sanitized;
}

/*
 * Sanitizes all string fields in a block object extracted from LLM output.
 */
function sanitizeBlock(block: any): any {
    if (typeof block === 'string') {
        return sanitizeLLMString(block);
    }
    if (typeof block === 'object' && block !== null) {
        const sanitized: any = {};
        for (const key of Object.keys(block)) {
            if (typeof block[key] === 'string') {
                sanitized[key] = sanitizeLLMString(block[key]);
            } else {
                sanitized[key] = block[key];
            }
        }
        return sanitized;
    }
    return block;
}

export function ChatBlock({text, mimeType, url} : {
    text?: string,
    mimeType?: string,
    url?: string
}) {
    let internalComponent = <></>
    if (text) {
        internalComponent = <span>{text}</span>
    } else if (mimeType && url) {
        if (mimeType.startsWith("audio")) {
            internalComponent = <audio controls={true} src={url} />
        } else if (mimeType.startsWith("video")) {
            internalComponent = <video controls width="250">
                <source src={url} type={mimeType} />
                Download the <a href={url}>video</a>
            </video>
        } else if (mimeType.startsWith("image")) {
            internalComponent = <img src={url} />
        }
    } else if (url) {
        internalComponent = <a href={url}>Link</a>
    }

    return (
        <p className="text-sm text-gray-200 pb-2">
            {internalComponent}
        </p>
    );
}

/*
 * Take a completion, which may be a string, JSON encoded as a string, or JSON object,
 * and produce a list of ChatBlock objects. This is intended to be a one-size-fits-all
 * method for funneling different LLM output into structure that supports different media
 * types and can easily grow to support more metadata (such as speaker).
 */
export function responseToChatBlocks(completion: any) {
    // First we try to parse completion as JSON in case we're dealing with an object.
    console.log("got completoin", completion, typeof completion)
    if (typeof completion == "string") {
        try {
            completion = JSON.parse(completion)
        } catch {
            // Do nothing; we'll just treat it as a string.
            console.log("Couldn't parse")
        }
    }
    let blocks = []
    if (typeof completion == "string") {
        console.log("still string")
        const sanitized = sanitizeLLMString(completion);
        blocks.push(<ChatBlock text={sanitized} />)
    } else if (Array.isArray(completion)) {
        console.log("Is array")
        for (let block of completion) {
            console.log(block)
            const sanitizedBlock = sanitizeBlock(block);
            blocks.push(<ChatBlock {...sanitizedBlock} />)
        }
    } else {
        const sanitizedBlock = sanitizeBlock(completion);
        blocks.push(<ChatBlock {...sanitizedBlock} />)
    }
    console.log(blocks)
    return blocks
}