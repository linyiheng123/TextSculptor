import base64
import importlib
import json
import random
import re
import time
from typing import Any, Dict, List, Optional

openai = importlib.import_module("openai")

from metrics.config import get_api_config


class TAVQMetric:
    """TA/VQ metric with full-image text alignment.

    Compared with V1:
    - expected_target_words: model-inferred full-image expected text
    - observed_target_words: full-image observed text from edited image
    """

    name = "ta_vq_score"
    supported_edit_types = ["replace", "remove", "add", "hybrid"]

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        config = get_api_config()
        self.key_list = config["keys"]
        self.azure_endpoint = config["azure_endpoint"]
        self.api_version = config["api_version"]
        self.model_name = config["model_name"]

    def build_system_prompt(self, edit_type: str = "replace") -> str:
        q1_definitions = {
            "replace": '''location_correct (Yes/No):
- This question is about text placement only.
- Answer "Yes" only if the new replacement text appears at the exact location of the text being replaced, as required by the instruction.
- Background / non-target text that should remain must keep its original position and layout, without noticeable shifting, drifting, reflow, or reordering.
- Answer "No" if the replacement is applied at the wrong location, appears offset or duplicated elsewhere, or if background / non-target text changes position noticeably.''',
            "add": '''location_correct (Yes/No):
- This question is about text placement only.
- Answer "Yes" only if the added text appears at the intended new location specified by the instruction.
- Background / non-target text that should remain must keep its original position and layout, without noticeable shifting, drifting, reflow, or reordering.
- Answer "No" if the added text is misplaced, attached to the wrong region, duplicated elsewhere, or if background / non-target text changes position noticeably.''',
            "remove": '''location_correct (Yes/No):
- This question is about text placement only.
- Answer "Yes" only if the intended target text is removed or cleared at the correct target location.
- Background / non-target text that should remain must keep its original position and layout, without noticeable shifting, drifting, reflow, or reordering.
- Answer "No" if the wrong text/region is removed, the target location is not correctly edited, or if background / non-target text changes position noticeably.''',
            "hybrid": '''location_correct (Yes/No):
- This question is about text placement only.
- Answer "Yes" only if each instructed edit is applied at its intended target location.
- Background / non-target text that should remain must keep its original position and layout, without noticeable shifting, drifting, reflow, or reordering.
- Answer "No" if any target edit is applied to the wrong location, any edited text appears offset or duplicated elsewhere, or if background / non-target text changes position noticeably.''',
        }

        q1_def = q1_definitions.get(edit_type, q1_definitions["replace"])

        return rf"""
You are a strict image editing quality judge for TEXT EDITING tasks in a TA/VQ format.

You will receive:
- Original image (before edit)
- Edited image (after edit)
- Editing instruction in natural language

Your job:
1) Visually compare ORIGINAL vs EDITED images (THIS IS NOT OCR).
2) Answer the visual-quality questions based ONLY on what you can SEE.
3) Output expected_target_words and observed_target_words for the WHOLE IMAGE text (not only target region).
4) Output ONLY valid JSON in the required schema. Do NOT output markdown. Do NOT include extra keys.

============================================================
EXPECTED VS OBSERVED
============================================================

- expected_target_words:
  Infer the complete text that SHOULD appear in the EDITED image after correctly applying the instruction to the ORIGINAL image.
  Include BOTH edited target text and all background text that should remain.
  Output as one space-separated word string in natural reading order (top-to-bottom, left-to-right).

- observed_target_words:
  Transcribe the complete text you can VISUALLY read from the EDITED image across the WHOLE image.
  Include both target and background text.
  Output as one space-separated word string in natural reading order (top-to-bottom, left-to-right).

============================================================
CRITICAL RULE (NO GUESSING FOR OBSERVED TEXT)
============================================================

- observed_target_words MUST be a literal transcription of what is VISUALLY LEGIBLE in the EDITED image.
- You MUST NOT use expected_target_words, instruction, context, or common sense to fill in unreadable words for observed_target_words.
- If a word is not clearly readable, you MUST output a placeholder token instead of guessing.

============================================================
PLACEHOLDER RULES (MANDATORY)
============================================================

- Use the exact placeholder token: [UNREADABLE]
- If a whole word is unreadable/blurred/occluded -> output [UNREADABLE] for that word.
- If only part of a word is readable -> output [UNREADABLE] (do NOT guess partials).
- Never replace [UNREADABLE] with a guessed word.
- If you output ANY [UNREADABLE] tokens:
  - Add an issue describing which part is unreadable.
  - If readability is unreliable, physical_plausibility should be "No".

============================================================
DEFINITIONS
============================================================

{q1_def}

Judge location_correct using these global rules as well:
- Focus only on placement; do not judge style, readability, or physical realism here.
- If the instruction implies a specific placement and the edited result does not follow that placement, answer "No".
- If background / non-target text moves, shifts, swaps order, or changes layout position noticeably, answer "No".

------------------------------------------------------------

style_consistent (Yes/No):

This question is ONLY about overall visual style identity.

Judge both:
- the edited target text
- any preserved background / non-target text that should have remained unchanged

Judge style by:
- text color
- font type vibe (serif vs sans-serif, handwritten vs printed)
- stroke weight (thin vs bold)
- presence of outline/shadow/glow

Answer "Yes" only if:
- the edited target text looks like the same style family as the original target text, and
- preserved background / non-target text does not show noticeable unintended style changes.

Answer "No" if ANY of the following happen:
- the edited target text has a clear style identity mismatch (e.g., different color or clearly different font type).
- preserved background / non-target text changes in color, font vibe, stroke weight, outline, shadow, glow, or other style identity.

If uncertain, answer "Yes".

------------------------------------------------------------

physical_plausibility (Yes/No):

Does the edited result look clear, physically plausible, and free of obvious editing artifacts, especially for the edited target text and any preserved background / non-target text affected by the edit?

Consider:
- text clarity / readability
- lighting consistency
- perspective consistency
- surface curvature conformity
- edge halos
- blending artifacts
- smearing / overpainting
- unnatural sharpness or blur mismatch

Also answer "No" if preserved background / non-target text is noticeably degraded by the edit, such as:
- becoming blurrier or less readable
- showing halos, ghosting, distortion, smearing, or artifacting
- looking unintentionally overpainted or damaged

IMPORTANT:
If text in edited regions or preserved background text is heavily blurred/garbled and NOT reliably readable -> answer "No".

============================================================
OUTPUT FORMAT (STRICT)
============================================================

Return ONLY valid JSON with EXACT keys:

{{
  "answers": {{
    "location_correct": "Yes/No",
    "style_consistent": "Yes/No",
    "physical_plausibility": "Yes/No",
    "expected_target_words": "<space-separated words for whole-image expected text>",
    "observed_target_words": "<space-separated words for whole-image observed text (may include [UNREADABLE])>"
  }},
  "issues": ["issue1", "issue2", ...]
}}

Do NOT include explanations outside JSON.
""".strip()

    def _normalize_word_token(self, w: str) -> str:
        if not isinstance(w, str):
            return ""

        w = w.strip()
        if w.strip(".,;:!?\"'(){}<>") == "[UNREADABLE]":
            return "[UNREADABLE]"

        return re.sub(r"[^a-zA-Z0-9]", "", w)

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            print(f"[Error] Encoding image {image_path}: {exc}")
            return None

    def _perform_gpt_judge(self, job: Dict[str, str], edit_type: str = "replace") -> Optional[Dict[str, Any]]:
        orig_b64 = self._encode_image_to_base64(job["original_image_path"])
        if not orig_b64:
            return None

        edited_b64 = self._encode_image_to_base64(job["edited_image_path"])
        if not edited_b64:
            return None

        system_prompt = self.build_system_prompt(edit_type)
        user_payload = [
            {"type": "text", "text": f"Instruction: {job['instruction']}"},
            {"type": "text", "text": "Image 1: ORIGINAL (before edit)"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
            {"type": "text", "text": "Image 2: EDITED (after edit)"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
            {"type": "text", "text": "Answer the questions and output JSON only."},
        ]

        max_retries = 8
        for attempt in range(1, max_retries + 1):
            api_key = random.choice(self.key_list)
            client = openai.AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                api_key=api_key,
            )
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.0,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as exc:
                error_text = str(exc)
                is_rate_limit = "429" in error_text or "rate limit" in error_text.lower()
                if is_rate_limit and attempt < max_retries:
                    sleep_seconds = random.uniform(20.0, 40.0)
                    print(
                        f"[RateLimit] Retry {attempt}/{max_retries} "
                        f"after {sleep_seconds:.1f}s: {job.get('filename', '')}"
                    )
                    time.sleep(sleep_seconds)
                    continue
                print(f"[API Error] {job.get('filename', '')}: {exc}")
                return None

        return None

    def tokenize_words(self, s: str) -> List[str]:
        if not isinstance(s, str):
            return []
        return [token for token in (x.strip() for x in s.strip().split()) if token]

    def align_word_errors(self, expected: List[str], observed: List[str]) -> Dict[str, Any]:
        exp = [self._normalize_word_token(w) for w in (expected or [])]
        obs = [self._normalize_word_token(w) for w in (observed or [])]
        exp = [w for w in exp if w]
        obs = [w for w in obs if w]

        n, m = len(exp), len(obs)
        dp: List[List[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            cost, subs, ins, dele = dp[i - 1][0]
            dp[i][0] = (cost + 1, subs, ins, dele + 1)
        for j in range(1, m + 1):
            cost, subs, ins, dele = dp[0][j - 1]
            dp[0][j] = (cost + 1, subs, ins + 1, dele)

        def better(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return a if a < b else b

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if exp[i - 1] == obs[j - 1]:
                    prev = dp[i - 1][j - 1]
                    cand_sub = (prev[0], prev[1], prev[2], prev[3])
                else:
                    prev = dp[i - 1][j - 1]
                    cand_sub = (prev[0] + 1, prev[1] + 1, prev[2], prev[3])

                prev = dp[i][j - 1]
                cand_ins = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])

                prev = dp[i - 1][j]
                cand_del = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)

                dp[i][j] = better(cand_sub, better(cand_ins, cand_del))

        cost, subs, ins, dele = dp[n][m]
        total_err = int(subs + ins + dele)
        expected_len = max(1, n)
        wer = total_err / expected_len

        return {
            "edit_distance": int(cost),
            "misspelled_words": int(subs),
            "extra_words": int(ins),
            "missing_words": int(dele),
            "total_word_errors": total_err,
            "word_error_rate": float(wer),
            "expected_word_count": int(n),
            "observed_word_count": int(m),
        }

    def _answer_to_binary_score(self, value: Any) -> float:
        if isinstance(value, str) and value.strip() == "Yes":
            return 1.0
        return 0.0

    def calculate(
        self,
        original_image_path: str,
        edited_image_path: str,
        edit_data: object,
    ) -> Dict[str, Any]:
        edit_type = getattr(edit_data, "edit_type", None)
        if edit_type not in self.supported_edit_types:
            return {"error": f"Unsupported edit type for ta_vq_score: {edit_type}"}

        job = {
            "filename": getattr(edit_data, "image_filename", ""),
            "original_image_path": original_image_path,
            "edited_image_path": edited_image_path,
            "instruction": edit_data.instruction,
        }

        raw = self._perform_gpt_judge(job, edit_type)
        if not raw or not isinstance(raw, dict):
            return {"error": "GPT judge failed"}

        answers = raw.get("answers", {}) if isinstance(raw.get("answers"), dict) else {}
        expected_str = answers.get("expected_target_words", "")
        observed_str = answers.get("observed_target_words", "")

        expected_tokens = self.tokenize_words(expected_str)
        observed_tokens = self.tokenize_words(observed_str)
        align_counts = self.align_word_errors(expected_tokens, observed_tokens)

        word_error_rate = float(align_counts.get("word_error_rate", 1.0))
        expected_edit_word_count = max(int(getattr(edit_data, "expected_edit_word_count", 0) or 0), 1)
        total_word_errors = float(align_counts.get("total_word_errors", 0.0))
        edit_word_error_rate = total_word_errors / float(expected_edit_word_count)
        text_accuracy_score = 1.0 - min(edit_word_error_rate, 1.0)

        visual_subscores = {
            "location_correct": self._answer_to_binary_score(answers.get("location_correct", "")),
            "style_consistent": self._answer_to_binary_score(answers.get("style_consistent", "")),
            "physical_plausibility": self._answer_to_binary_score(answers.get("physical_plausibility", "")),
        }
        visual_quality_score = sum(visual_subscores.values()) / len(visual_subscores)

        visual_answers = [
            f"location={answers.get('location_correct', '')}",
            f"style={answers.get('style_consistent', '')}",
            f"physical={answers.get('physical_plausibility', '')}",
        ]
        reason = (
            "text_alignment("
            f"sub={align_counts['misspelled_words']}, "
            f"ins={align_counts['extra_words']}, "
            f"del={align_counts['missing_words']}, "
            f"total={align_counts['total_word_errors']}, "
            f"wer={word_error_rate:.4f}, "
            f"edit_words={expected_edit_word_count}, "
            f"edit_error_rate={edit_word_error_rate:.4f}, "
            f"text_accuracy={text_accuracy_score:.4f}"
            "); visual(" + ", ".join(visual_answers) + ")"
        )

        issues_obj = raw.get("issues", [])
        issues: List[Any] = issues_obj if isinstance(issues_obj, list) else []

        return {
            "text_accuracy_score": text_accuracy_score,
            "visual_quality_score": visual_quality_score,
            "visual_subscores": visual_subscores,
            "reason": reason,
            "answers": {
                "location_correct": answers.get("location_correct", ""),
                "style_consistent": answers.get("style_consistent", ""),
                "physical_plausibility": answers.get("physical_plausibility", ""),
                "expected_target_words": expected_str,
                "observed_target_words": observed_str,
            },
            "align_counts": align_counts,
            "expected_edit_word_count": expected_edit_word_count,
            "edit_word_error_rate": edit_word_error_rate,
            "issues": issues,
        }
