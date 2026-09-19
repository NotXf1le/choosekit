import { isCount, ScoringError } from "./validation.js";

export function tokenIds(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new ScoringError("Expected a nonempty array of token IDs.");
  }
  const ids: number[] = [];
  for (const id of value) {
    if (!isCount(id)) {
      throw new ScoringError("Token IDs must be nonnegative safe integers.");
    }
    ids.push(id);
  }
  return ids;
}

export interface TokenNode {
  readonly children: Map<number, TokenNode>;
  index?: number;
}

export function candidateTree(prefix: readonly number[], alternatives: readonly number[][]): {
  root: TokenNode; shared: number;
} {
  let shared = prefix.length;
  for (const ids of alternatives) {
    if (ids.length === prefix.length && ids.every((id, i) => id === prefix[i])) {
      throw new ScoringError("A candidate disappeared during tokenization.");
    }
    shared = Math.min(shared, ids.length);
    for (let i = 0; i < shared; i++) {
      if (ids[i] !== prefix[i]) { shared = i; break; }
    }
  }
  if (shared === 0) throw new ScoringError("No stable token prefix; use a model-appropriate prompt.");
  const root: TokenNode = { children: new Map() };
  for (let index = 0; index < alternatives.length; index++) {
    const ids = alternatives[index]!;
    if (ids.length <= shared) throw new ScoringError("A candidate has no tokens after the shared prefix.");
    let node = root;
    for (let position = shared; position < ids.length; position++) {
      if (node.index !== undefined) throw new ScoringError("Candidate token sequences overlap.");
      const id = ids[position]!;
      let child = node.children.get(id);
      if (!child) {
        child = { children: new Map() };
        node.children.set(id, child);
      }
      node = child;
    }
    if (node.index !== undefined) throw new ScoringError("Different candidates have the same token sequence.");
    if (node.children.size !== 0) throw new ScoringError("Candidate token sequences overlap.");
    node.index = index;
  }
  return { root, shared };
}
