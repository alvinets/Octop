/** Harness browser tools that activate the in-chat browser workspace. */
export const BROWSER_TOOL_NAMES = ["browser_use", "browser_control"] as const;

/** Harness tools that write into the agent workspace and open the edit-file card. */
export const FILE_TOOL_NAMES = [
  "write_file",
  "edit_file",
  "send_file",
  "send_file_to_user",
] as const;

/**
 * Tools whose completion can add thread artifacts, which is a wider set than
 * {@link FILE_TOOL_NAMES}: a shell command can leave a new file behind even
 * though it is not itself a file-edit tool. Kept separate so the dock refetches
 * after any of them without also counting them as "files edited this turn",
 * which would inflate the file count and surface an edit affordance on every
 * shell call.
 */
export const ARTIFACT_TOOL_NAMES = [
  ...FILE_TOOL_NAMES,
  "execute",
  "execute_shell_command",
] as const;

export const EMPTY_CHAT_SESSION_KEY = "__empty__";
export const PENDING_THREAD_ID = "__pending__";

function toolNameBase(name: string): string {
  const trimmed = name.trim();
  const slash = trimmed.lastIndexOf("/");
  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed;
}

export function isBrowserToolName(name: string | undefined): boolean {
  return (BROWSER_TOOL_NAMES as readonly string[]).includes(name ?? "");
}

/** Match ``write_file`` / ``edit_file`` / ``send_file`` (and ``ns/…``), not ``write_todos``. */
export function isFileToolName(name: string | undefined): boolean {
  const base = toolNameBase((name ?? "").trim()).toLowerCase();
  return (FILE_TOOL_NAMES as readonly string[]).includes(base);
}

/** Alias kept for call sites that mean "opens the chat file panel card". */
export function isWriteToolName(name: string | undefined): boolean {
  return isFileToolName(name);
}

/** True when completing this tool can add thread artifacts (superset of {@link isFileToolName}). */
export function isArtifactToolName(name: string | undefined): boolean {
  const base = toolNameBase((name ?? "").trim()).toLowerCase();
  return (ARTIFACT_TOOL_NAMES as readonly string[]).includes(base);
}
