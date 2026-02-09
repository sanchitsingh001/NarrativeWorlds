"""
overview_story_generator.py

Goal:
- Take a raw user story idea (free-form text)
- Convert it into a premium, world-first, OUTDOOR-ONLY narrative overview
- Output: a single prose string suitable to pass into Worldplan.py as `story_text`

Requires:
    pip install openai>=1.0.0

Auth:
- ✅ Reads your DeepInfra token from environment variables:
    - DEEPINFRA_API_KEY (preferred)
    - DEEPINFRA_TOKEN (fallback)
"""

import os
from openai import OpenAI

# -----------------------------
# 1) CONFIG (env var auth)
# -----------------------------
MODEL = "deepseek-ai/DeepSeek-V3.2"

def get_deepinfra_api_key() -> str:
    """
    Read token from environment. Prefer DEEPINFRA_API_KEY, fallback to DEEPINFRA_TOKEN.
    """
    key = (os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPINFRA_TOKEN") or "").strip()
    if not key:
        raise RuntimeError(
            "Missing DeepInfra API key. Set environment variable DEEPINFRA_API_KEY "
            "(or DEEPINFRA_TOKEN as fallback).\n\n"
            "Example:\n"
            "  export DEEPINFRA_API_KEY='your_token_here'\n"
        )
    return key

# Your raw story idea (free-form). Replace this with any story you want.
USER_STORY = r"""
Everyone in My Hometown Acts Like They Know Me. I Don’t Remember Any of Them.

When I came back to town, everyone treated me like I’d never left.

That should have felt comforting. It didn’t.

The bus dropped me off at the old stop near the square. Same cracked pavement. Same crooked lamppost. The air smelled like cut grass and bread from the bakery down the street. I remember thinking how strange it was that everything looked exactly the way it had when I was a kid, even though I knew that couldn’t be true.

An old man was standing by the gate that led into the neighborhood. He leaned on the iron bars like he’d been waiting for me.

“Back so soon, Elias?” he said, smiling. “The garden hasn’t been the same since you tidied it up.”

I didn’t know his face. I didn’t recognize his voice.

“Who are you?” I asked.

He laughed. A warm, familiar sound. “Always joking. Just like your father. Even before the accident.”

Something about the way he said the accident made my stomach tighten.

I asked him where my house was.

He pointed down the street without hesitation.

Everyone did.

At the bakery, Sarah was already wrapping a loaf when I walked in.

“I made your favorite,” she said. “No poppy seeds this time. We know how you get when you see small black dots.”

I stared at her. I don’t remember telling anyone that. I don’t even remember it being true.

She slid the loaf across the counter.

It wasn’t bread.

It was gray. Damp-looking. Slightly misshapen. When I touched it, it felt warm. Soft. It pulsed once under my fingers, like something alive.

I pulled my hand back. Sarah kept smiling like nothing was wrong.

As I walked farther into town, things began to feel off in ways I couldn’t explain. Buildings looked flatter than they should have. Like sets built for a play. The corners were too sharp. The walls too smooth. When I leaned against one storefront, it flexed slightly under my weight.

People passed me on the sidewalk and muttered things under their breath.

Things I had been thinking.

“You should have stayed,” one voice said.

“You never finish what you start,” said another.

At the school, the bell rang as I stepped inside. A woman stood at the front of a classroom.

“Class is waiting, Elias,” she said. “Why are you late?”

“I’m not a student,” I said. “What class?”

She turned around.

Her face was a mirror.

I saw myself staring back, pale and hollow-eyed.

“The class on forgetting,” she whispered.

The sound in town changed after that. The gentle piano music that seemed to come from nowhere slowed. Deepened. It became a low, rhythmic thumping that I felt more than heard.

Like a heartbeat.

Posters appeared on the walls where old flyers had been. Where there had once been notices about missing pets and church events, there were now only two kinds of posters.

Missing Person.

Found: Elias.

The photo underneath showed a coffin.

I reached my house just as the sun began to set. It was the only building that looked real. Solid. Worn. Familiar in a way nothing else had been.

A man stood in the doorway.

I knew he was a doctor before he spoke.

“You’ve done a full lap,” he said calmly. “Do you remember why we built this place for you?”

I wanted to say no. I wanted to run.

Instead, the words came out on their own.

“I killed them, didn’t I?”

He didn’t argue.

“This isn’t a town,” he said. “It’s a memory palace. You’re not walking streets. You’re pacing your cell.”

The truth came back in pieces. A fire. Or a crash. Smoke. Screaming. Hands slipping out of mine.

The people here weren’t strangers. They were the ones I couldn’t save.

I went inside my house.

There was no furniture. No pictures. Just a white room and a single chair in the center.

A child sat on the floor nearby. I couldn’t tell if I recognized them or not.

“Don’t leave again,” the child said softly. “It’s cold when you forget us.”

The room began to fade. Not into darkness. Into white.

My legs felt heavy. Each step slower than the last. Like my body was remembering something my mind didn’t want to.

The last thing I saw before I couldn’t move anymore was the chair waiting for me.
""".strip()

TONE_HINT = "psychological horror"

# -----------------------------
# 2) PROMPT (world-first)
# -----------------------------
SYSTEM_PROMPT = f"""
You are a Narrative Preprocessor for a procedural world generator.

TASK:
Rewrite the user's raw story idea into an "enhanced story overview" that will be used as input to a world-planning model.

HARD CONSTRAINTS (must follow):
- Output MUST be plain text (no JSON, no bullet lists, no headings).
- Focus on WALKABLE EXTERIOR/OUTDOOR spaces only.
  Do NOT describe indoor rooms as separate locations. If interior is implied, mention it as something the player never fully sees.
- Make the world planner's job easy:
  Mention 4–6 distinct OUTDOOR areas implicitly through description (examples: clearing, tower exterior, stair base, trail junction, service road, ranger gate).
- Keep it "walk and talk" horror, but DO NOT enumerate NPC specs.
  NPCs can be mentioned as part of the story naturally (as voices encountered), but don't output a cast list.
- Prioritize architectural/exterior details and navigable landmarks:
  materials, shapes, silhouettes, signage, fences, radios, stairs, trapdoor outline, lookout cabin exterior, etc.
- Tone: {TONE_HINT}. Premium indie psychological horror. Fresh metaphors, escalating wrongness. No clichés.

LENGTH:
- 12–18 sentences total. Present tense. Second person ("you").
- Clear arc: setup → cracks → escalation → twist → exit sting.
Return ONLY the rewritten story.
""".strip()

# -----------------------------
# 3) CALL DEEPINFRA
# -----------------------------
def generate_enhanced_story(user_story: str) -> str:
    client = OpenAI(
        api_key=get_deepinfra_api_key(),
        base_url="https://api.deepinfra.com/v1/openai",
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_story},
        ],
        temperature=0.9,
    )
    return (resp.choices[0].message.content or "").strip()

# -----------------------------
# 4) RUN + PRINT RESULT
# -----------------------------
if __name__ == "__main__":
    enhanced_story_text = generate_enhanced_story(USER_STORY)

    print("\n\n===== ENHANCED STORY (paste into Worldplan.py `story = '''...'''`) =====\n")
    print(enhanced_story_text)
    print("\n\n===== END =====\n")
