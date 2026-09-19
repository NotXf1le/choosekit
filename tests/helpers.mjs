export const request = Object.freeze({
  context: "The implementation changed; tests have not run.",
  question: "What should happen next?",
  choices: Object.freeze({ edit: "Modify the code.", test: "Run the tests.", done: "Finish." }),
});

export function encode(text, special = false) {
  const ids = [...new TextEncoder().encode(text)].map((byte) => byte + 1);
  return special ? [0, ...ids] : ids;
}

export function random(seed = 42) {
  return () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
}
