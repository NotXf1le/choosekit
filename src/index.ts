import { createFormattedChooser } from "./internal-chooser.js";
import type { Chooser, ChooserOptions, Scorer } from "./types.js";

export type {
  Choices, ChoiceKey, ChoiceRequest, Chooser, ChooserOptions, Decision,
  ImageInput, ImageMediaType, ScoreRequest, Scorer, Scores, Usage, PromptInput,
} from "./types.js";
export { ScoringError } from "./validation.js";

export function createChooser(score: Scorer, options: ChooserOptions = {}): Chooser {
  return createFormattedChooser(score, options, "keys");
}
