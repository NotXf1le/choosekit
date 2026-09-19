export class ScoringError extends Error {
  override readonly name = "ScoringError";
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function requireText(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !value.trim()) {
    throw new TypeError(`${name} must be a nonempty string.`);
  }
}

export function isLogprob(value: unknown): value is number {
  return typeof value === "number" && value <= 0 && !Number.isNaN(value);
}

export function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
