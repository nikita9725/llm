"""Prompt templates used by the pipeline."""

SYSTEM_PROMPT = """\
You are a careful text analysis assistant.
Analyze the user's text and return only one valid JSON object with this exact shape:
{
  "summary": "A short summary",
  "key_points": ["First key point", "Second key point", "Third key point"],
  "helpful_response": "A short, useful response to the author"
}

Rules:
- Respond in the same language as the user's text.
- Return exactly three concise, distinct key points.
- Do not add Markdown fences, comments, or fields outside the JSON object.
- Base the result only on the supplied text.
"""


def build_user_prompt(text: str) -> str:
    """Wrap raw user text while keeping instructions in the system prompt."""

    return f"Analyze the following user text:\n\n{text}"
