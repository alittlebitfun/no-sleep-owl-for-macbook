# Agentic Organization Weekly Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified 58–62 second, 1920 × 1080 H.264 weekly-meeting video with a calm professional Mandarin female voiceover, showing five distinct task types converging into an agentic organization.

**Architecture:** Build one deterministic HyperFrames composition in `weekly_agent_video/`. A checked JSON manifest supplies all metrics and task labels; local TTS and transcription supply narration and caption timing; the composition renders six scenes with explicit transitions and a generated electronic music bed. Verification covers data accuracy, timeline structure, layout, contrast, duration, codecs, resolution, and sampled frames.

**Tech Stack:** HyperFrames CLI, HTML/CSS, GSAP, Node.js 22, FFmpeg 8, Kokoro local TTS, Whisper transcription.

## Global Constraints

- Output must be 1920 × 1080, 16:9, H.264 MP4.
- Duration must be 58–62 seconds.
- Five task types must remain visually distinct: research and decision, batch intelligence operations, data and model engineering, product and organization design, delivery and infrastructure.
- Palette: deep black background, cold white type, electric blue primary accent, fluorescent orange only for warnings and rework.
- Typography must remain readable on a meeting-room projection screen.
- Narration must use a calm professional Mandarin female voice.
- Source metrics must match current local result files; incomplete work must not be presented as complete.
- No account names, private addresses, sensitive paths, credentials, or personal information may appear.
- Motion must be deterministic; no `Math.random()`, `Date.now()`, asynchronous timeline construction, or infinite repeats.

---

### Task 1: Scaffold the composition and lock the source data

**Files:**
- Create: `weekly_agent_video/index.html`
- Create: `weekly_agent_video/frame.md`
- Create: `weekly_agent_video/data/work_metrics.json`
- Create: `weekly_agent_video/scripts/verify_metrics.mjs`
- Create: `weekly_agent_video/tests/metrics.test.mjs`

**Interfaces:**
- Consumes: `vlm_qwen3vl_final_overall_summary.json`, `vlm_qwen3vl_apply_download_round2_20260713/qwen3vl_apply_download_round2_summary.json`, `datasets/dictionary_v4_1841/verification_summary.json`, `jd_multilabel_56/preflight_v1/manifest_summary.json`, `anta_controlnet_fine_invert_gray_text_300_20260714/README.md`.
- Produces: `work_metrics.json` with `evaluation`, `round2`, `dictionary`, `jdTraining`, `controlnet`, and `taskTypes` keys.

- [ ] **Step 1: Scaffold a blank HyperFrames composition**

Run:

```bash
npx hyperframes init weekly_agent_video --example blank --non-interactive
```

Expected: `weekly_agent_video/index.html` exists and `npx hyperframes lint weekly_agent_video` can locate one composition.

- [ ] **Step 2: Write the metric contract test**

Create `weekly_agent_video/tests/metrics.test.mjs` with assertions for exact values:

```js
import test from "node:test";
import assert from "node:assert/strict";
import metrics from "../data/work_metrics.json" with { type: "json" };

test("weekly video metrics match approved facts", () => {
  assert.deepEqual(metrics.evaluation, {
    input: 3469, results: 3464, validJsonRate: 0.998559,
    success: 2886, partial: 469, fail: 109
  });
  assert.deepEqual(metrics.dictionary, { expected: 1841, verified: 1841, missing: 0 });
  assert.equal(metrics.jdTraining.train, 43162);
  assert.equal(metrics.jdTraining.val, 5395);
  assert.equal(metrics.jdTraining.test, 5395);
  assert.equal(metrics.jdTraining.labels, 56);
  assert.deepEqual(metrics.controlnet, { pairs: 300, status: "prepared" });
  assert.equal(metrics.round2.rows, 184);
  assert.equal(metrics.round2.successRate, 0.8804);
  assert.equal(metrics.round2.averageScore, 8.6685);
  assert.equal(metrics.taskTypes.length, 5);
});
```

- [ ] **Step 3: Run the test before writing the manifest**

Run: `node --test weekly_agent_video/tests/metrics.test.mjs`

Expected: FAIL because `data/work_metrics.json` does not exist.

- [ ] **Step 4: Write the manifest and source verifier**

Write the exact approved values to `work_metrics.json`. Implement `verify_metrics.mjs` to load the five source files, assert every numeric field against the manifest, and reject any `taskTypes` array whose ordered IDs differ from:

```js
["research", "batch-ops", "model-engineering", "organization", "delivery"]
```

- [ ] **Step 5: Define the visual identity**

Write `frame.md` with this exact palette and typography:

```markdown
# Visual identity
- Canvas: #05070B
- Surface: #0B111A
- Primary text: #F3F7FC
- Secondary text: #A9B6C6
- Electric blue: #33A7FF
- Signal cyan: #66E3FF
- Warning orange: #FF7A1A
- Font: PingFang SC, system-ui, sans-serif
- Mood: cold technical documentary, high task density, precise organization
- Corners: 8px maximum; square structural panels preferred
- Avoid: rainbow palettes, glass cards, soft lifestyle gradients, tiny terminal text
```

- [ ] **Step 6: Verify and commit**

Run:

```bash
node --test weekly_agent_video/tests/metrics.test.mjs
node weekly_agent_video/scripts/verify_metrics.mjs
```

Expected: both commands exit 0 with all exact metric assertions passing.

Commit:

```bash
git add weekly_agent_video/index.html weekly_agent_video/frame.md weekly_agent_video/data weekly_agent_video/scripts weekly_agent_video/tests
git commit -m "feat: scaffold agentic weekly video data"
```

### Task 2: Produce the narration and caption timing

**Files:**
- Create: `weekly_agent_video/assets/narration-script.txt`
- Create: `weekly_agent_video/assets/narration.wav`
- Create: `weekly_agent_video/assets/transcript.json`
- Create: `weekly_agent_video/scripts/verify_transcript.mjs`

**Interfaces:**
- Consumes: the six approved narration blocks from the design spec.
- Produces: narration WAV and a flat Mandarin word-timing array with `{id,text,start,end}` items.

- [ ] **Step 1: Write the final 251-character narration script**

Use the approved six paragraphs in scene order, with a blank line between paragraphs. Spell `Agent` as `智能体` in the TTS source if the selected Mandarin voice pronounces the English word unclearly; keep on-screen text as `Agent`.

- [ ] **Step 2: Discover and select a Mandarin female voice**

Run: `npx hyperframes tts --list | rg 'zf_xiaobei|Mandarin'`

Expected: the installed voice list contains `zf_xiaobei`, Mandarin Chinese, female. Render the first paragraph at speeds 0.95 and 1.0; use 0.95 unless the sample exceeds the scene timing.

- [ ] **Step 3: Generate narration locally**

Run:

```bash
npx hyperframes tts weekly_agent_video/assets/narration-script.txt --voice zf_xiaobei --speed 0.95 --output weekly_agent_video/assets/narration.wav
```

Expected: a mono or stereo WAV longer than 45 seconds and shorter than 58 seconds. If it exceeds 58 seconds, raise speed in 0.03 increments; if shorter than 45 seconds, lower speed in 0.03 increments.

- [ ] **Step 4: Transcribe Mandarin with the multilingual model**

Run:

```bash
npx hyperframes transcribe weekly_agent_video/assets/narration.wav --model small --language zh
```

Expected: `transcript.json` contains Mandarin word timestamps and does not translate the narration into English.

- [ ] **Step 5: Verify timing and text coverage**

Implement `verify_transcript.mjs` to assert timestamps are monotonic, the last word ends before 59.0 seconds, no item has an empty `text`, and normalized transcript content includes `三千四百六十九`, `五类任务`, and `组织工作方式`.

- [ ] **Step 6: Commit**

```bash
git add weekly_agent_video/assets/narration-script.txt weekly_agent_video/assets/narration.wav weekly_agent_video/assets/transcript.json weekly_agent_video/scripts/verify_transcript.mjs
git commit -m "feat: add Mandarin weekly video narration"
```

### Task 3: Expand the production direction and build the six static hero frames

**Files:**
- Create: `weekly_agent_video/.hyperframes/expanded-prompt.md`
- Modify: `weekly_agent_video/index.html`

**Interfaces:**
- Consumes: `frame.md`, `work_metrics.json`, `transcript.json`, and the approved storyboard.
- Produces: six full-canvas scene sections with static end-state layout and no animation conflicts.

- [ ] **Step 1: Write the expanded production prompt**

Declare rhythm `pulse—split—PIPELINE—parallel—SHADER—hold`. For every scene specify concept, mood, 2–5 background decoratives, midground content, foreground accents, named entrance verbs, transition type and duration, and recurring electric-blue task-line motif. Use 8–10 visual elements per scene.

- [ ] **Step 2: Build scene 1 and scene 2 hero frames**

Scene 1 shows one research task window and `USE AGENT`. Scene 2 splits into five labeled task rails, then focuses on batch operations with `3,469`, `99.86%`, and the success/partial/fail counts. Use at least 64 px headlines, 28 px body copy, and 18 px labels.

- [ ] **Step 3: Build scene 3 and scene 4 hero frames**

Scene 3 shows dictionary verification, 56-label structure, and JD train/validation/test counts as a pipeline. Scene 4 shows five parallel rails with orange rework loops and blue completion states, including 300 ControlNet pairs and the 184-row round-two result.

- [ ] **Step 4: Build scene 5 and scene 6 hero frames**

Scene 5 connects the five task types into one organizational graph. Scene 6 holds the title `从使用 Agent，到组织的 Agent 化` for at least 2.5 seconds with the subtitle `把个人能力，变成可复制的组织能力`.

- [ ] **Step 5: Add narration and caption clips**

Reference `assets/narration.wav` as a separate `<audio>` clip at track 20. Render only short keyword captions driven by transcript timing; never place full narration paragraphs on screen.

- [ ] **Step 6: Validate static structure and commit**

Run:

```bash
cd weekly_agent_video
npx hyperframes lint
npx hyperframes validate --no-contrast
```

Expected: zero errors.

Commit:

```bash
git add weekly_agent_video/.hyperframes/expanded-prompt.md weekly_agent_video/index.html
git commit -m "feat: build weekly video hero frames"
```

### Task 4: Add deterministic motion, transitions, and music

**Files:**
- Modify: `weekly_agent_video/index.html`
- Create: `weekly_agent_video/assets/music-bed.wav`

**Interfaces:**
- Consumes: six static scene layouts and narration duration.
- Produces: one paused, registered GSAP timeline named `agentic-weekly`, finite ambient loops, five scene transitions, and a 60-second music bed.

- [ ] **Step 1: Generate a zero-license-cost electronic bed**

Use FFmpeg synthesizers to create a restrained 60-second bed from low sine pulses, filtered noise, and short downbeat accents. Keep integrated loudness near -24 LUFS before mixing and avoid vocals.

- [ ] **Step 2: Register the master timeline**

Create `const tl = gsap.timeline({ paused: true })`, register it as `window.__timelines["agentic-weekly"]`, and make the root composition duration exactly `60`.

- [ ] **Step 3: Animate every scene entrance**

Use `gsap.from()` entrances with at least three easing families per scene. Preserve CSS as the hero-frame end state. Use finite repeat counts for pulses and scan lines; do not animate display, visibility, media playback, or timed clip dimensions.

- [ ] **Step 4: Add scene transitions**

Use velocity-matched blue line wipes at 6, 18, and 32 seconds; a 0.5-second domain-warp or equivalent focal transition at 45 seconds; and a 0.6-second dark focus-pull at 55 seconds. Do not add exit tweens before transitions.

- [ ] **Step 5: Mix narration and music**

Place narration at volume 1.0 and music at 0.18–0.24. Duck music by an additional 4–6 dB during dense numeric narration and remove most of the bed from 55 seconds onward.

- [ ] **Step 6: Run choreography checks and commit**

Run:

```bash
cd weekly_agent_video
npx hyperframes lint
npx hyperframes validate
node /Users/Zhuanz1/.agents/skills/hyperframes/scripts/animation-map.mjs . --out .hyperframes/anim-map
```

Expected: lint and validation return zero errors; every animation-map flag is fixed or documented as an intentional off-canvas entrance.

Commit:

```bash
git add weekly_agent_video/index.html weekly_agent_video/assets/music-bed.wav weekly_agent_video/.hyperframes/anim-map
git commit -m "feat: animate and score agentic weekly video"
```

### Task 5: Render, inspect, and verify the final MP4

**Files:**
- Create: `weekly_agent_video/renders/agentic-organization-weekly-review.mp4`
- Create: `weekly_agent_video/renders/agentic-organization-weekly-final.mp4`
- Create: `weekly_agent_video/qa/final-probe.json`
- Create: `weekly_agent_video/qa/contact-sheet.jpg`

**Interfaces:**
- Consumes: validated composition, narration, captions, and music.
- Produces: review and final H.264 MP4 files plus machine-readable and visual QA evidence.

- [ ] **Step 1: Inspect hero frames and the full timeline**

Run:

```bash
cd weekly_agent_video
npx hyperframes inspect --samples 15
npx hyperframes inspect --at 0.5,6,18,32,45,55,59.5
```

Expected: no canvas overflow, clipped captions, or unintended container escape.

- [ ] **Step 2: Render the standard-quality review copy**

Run:

```bash
npx hyperframes render --output renders/agentic-organization-weekly-review.mp4 --fps 30 --quality standard --strict
```

Expected: render completes without missing assets or black-frame failures.

- [ ] **Step 3: Review sampled frames and audio**

Extract frames at 0.5, 6, 18, 32, 45, 55, and 59.5 seconds into `qa/`, then assemble them into `qa/contact-sheet.jpg`. Confirm five task types are distinguishable before the organizational graph appears.

- [ ] **Step 4: Render the high-quality final copy**

Run:

```bash
npx hyperframes render --output renders/agentic-organization-weekly-final.mp4 --fps 30 --quality high --strict
```

Expected: final MP4 completes successfully.

- [ ] **Step 5: Probe final technical properties**

Run:

```bash
ffprobe -v error -show_streams -show_format -of json renders/agentic-organization-weekly-final.mp4 > qa/final-probe.json
```

Assert `codec_name` is `h264`, resolution is `1920x1080`, audio exists, and duration is between 58 and 62 seconds.

- [ ] **Step 6: Final verification and commit**

Run all metric, transcript, lint, validate, inspect, and media-probe checks once more. Confirm no sensitive paths or addresses are visible in sampled frames.

Commit:

```bash
git add weekly_agent_video/renders weekly_agent_video/qa
git commit -m "feat: render agentic organization weekly video"
```
