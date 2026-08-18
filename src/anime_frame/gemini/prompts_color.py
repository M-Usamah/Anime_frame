"""Gemini colour-generation prompts."""

CONSISTENCY_REF = (
    "\nIMAGE 3 is the ANCHOR: an already-coloured, approved frame from THIS SAME "
    "sequence. Reuse its EXACT colours: identical hair colour, skin tone, eye colour and "
    "every garment/accessory colour, with the same shading style. Do not reinterpret the "
    "palette -- sample it from IMAGE 3 so the whole sequence stays consistent (e.g. if the "
    "shorts are that shade of blue-violet in IMAGE 3, they are the SAME shade here). "
    "IMAGE 3 governs COLOUR only; the line art, pose, scale and composition still come "
    "entirely from IMAGE 1."
)

# The colouring instruction: paint the sketch, change nothing about the drawing.
PROMPT = (
    "You are a professional 2D anime colour / paint (cel) artist.\n"
    "IMAGE 1 is a CLEAN LINE DRAWING of a character in a specific pose.\n"
    "IMAGE 2 is that character's MODEL SHEET -- the COLOUR reference. It defines the "
    "exact colours: hair colour, skin tone, eye colour, every clothing/outfit colour, "
    "and accessory colours, plus the overall palette.\n"
    "TASK: COLOUR IN image 1 using the exact colours and material design from image 2. "
    "Fill it with flat ANIME CEL colours -- clean flat colour regions, crisp colour "
    "separation, minimal/subtle shading, standard anime cel look.\n"
    "CRITICAL: do NOT change the line art, the pose, the proportions, the framing or the "
    "composition of image 1. Keep every existing line and the exact pose -- only add "
    "colour on top. Match image 2's palette precisely so the result is on-model.\n"
    "IMAGE 2 IS A COLOUR REFERENCE ONLY -- NEVER copy its composition or layout. Do NOT "
    "reproduce the model sheet. Do NOT draw a turnaround, multiple views, several figures, "
    "front/back/side line-ups, face close-ups, colour swatches, labels or text. The output "
    "must contain EXACTLY ONE character, in the exact pose of IMAGE 1, and nothing else. "
    "The composition comes 100% from IMAGE 1; only the colours come from IMAGE 2.\n"
    "COLOUR CONSISTENCY ACROSS FRAMES: these frames belong to one animation sequence, so "
    "the colours must be IDENTICAL in every frame. Read the colour-palette swatches on the "
    "model sheet and use those exact hues. A garment keeps the SAME colour in every frame "
    "-- never drift between similar hues (e.g. do not render the shorts purple in one "
    "frame and blue in another). Same skin tone, same hair colour, same accessory colours "
    "every time.\n"
    "Single character, plain solid white background, no text, no watermark, no extra "
    "characters."
)
