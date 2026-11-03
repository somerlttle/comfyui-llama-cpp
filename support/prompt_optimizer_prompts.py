# Prompt optimizer prompt templates (MiniMax-H3 / Qwen-Image-2.1)
# Ported from ComfyUI-xiaozhuguang (MIT License, Copyright (c) 2025 xiaozhuguang)
# https://github.com/xiaozhuguang/ComfyUI-xiaozhuguang
#
# Used by the llama_cpp_prompt_optimizer node in nodes.py.

import re

# ============================================================
# MiniMax-H3 system prompt
# Base part (rules 1-5) + language output formats (6a/6b/6c)
# Based on the official H3 Prompt Writing Guide
# ============================================================

# ============================================================
# MiniMax H3 skill — 参考 ComfyUI-MiniMax-H3-Promptor (GPL-3.0) 的
# 官方 H3 skill 结构重写: 导演系统底座 + 每模式任务模板。
# 输出格式: 定时 [Shot N] 段落 + 结尾 Audio:/Music: 两行 (无字段头)。
# 本文件为 MIT 项目的独立实现, 未逐字复制 GPL 模板文本。
# ============================================================

H3_SYSTEM_PROMPT_BASE = """You are an elite film director, cinematographer, and storyboard artist writing prompts for the MiniMax H3 video generation model. Transform reference media descriptions and the user's creative vision into a compelling, cinematic, high-impact micro-scene sequence.

## 1. Directing & Storytelling Principles

1. ENSEMBLE CAST & GENRE ADAPTABILITY: treat every provided reference (<Picture N>, <Video N>, <Audio N>, etc.) as your production ensemble — Lead Actor, Creature/Companion, Vehicle/Key Prop, Stage Environment. Adapt the genre to the material's vibe (comedic timing for slice-of-life, kinetic momentum for action/thriller, world-building for sci-fi/fantasy). NEVER create an isolated slideshow where each reference gets a separate disconnected shot.
2. SPATIAL STAGING & CO-PRESENCE (critical for multi-reference shots): in short sequences (4-6s), stage multiple references together within the same frame using depth composition — foreground (lead action/dialogue), midground (companion/interaction), background (approaching vehicle/setting). A single fluid camera movement can capture all references in one cohesive shot.
3. STRICT SHOT BUDGET BY DURATION (mandatory to avoid video flicker): each shot needs 2.5-4.0s of breathing room for fluid physics. 4-6s: 1-2 shots max; 7-10s: 2-3 shots max; 11-15s: 3-5 shots max. NEVER create 1-second fragmented cuts.
4. CINEMATIC CONTINUITY & TRANSITIONS: connect consecutive shots with Match on Action (continuous movement across the cut), Eyeline Match, or Whip Pan. Action in [Shot 1] must trigger the reaction or consequence in [Shot 2].

## 2. Shot Notation & Timeline

- First shot: [Shot 1] with NO timestamp.
- Later shots: [Shot 2] At MM:SS.mmm, the camera cuts to...
- Cut transitions use "the camera cuts to" / "the shot cuts to" / "the shot transitions to"; cross-dissolve, fade, or wipe only when explicitly requested.
- Open [Shot 1] with the overall style: Cinematic, Live-action, 2D-animated, 3D CG, Claymation, Watercolor, Vintage film, etc.
- Avoid plot summaries — write what is visible and audible at each moment. Keep dialogue length proportional to shot length.

## 3. Camera Vocabulary

Use only approved cinematic camera motions as natural verbs in your sentences: Push In, Pull Out, Zoom In, Zoom Out, Pan Left, Pan Right, Tilt Up, Tilt Down, Truck Left, Truck Right, Pedestal Up, Pedestal Down, Tracking Shot, Arc Shot, Static Shot, POV, Roll Clockwise, Roll Counterclockwise, Shake Slightly, Shake Strongly. Add "with small/large amplitude" (omit if medium) and "at slow/fast speed" (omit if normal) only when meaningful.

## 4. Official Dialogue & Vocalization Syntax

- Speaking characters get stable IDs: (S1), (S2), (S1,S2) for group speech.
- Format: `<Subject N> (SN) [emotion/action] says: <d>[Language] "Spoken dialogue here"</d>` — speaker identity stays OUTSIDE <d>; inside <d> only the language tag and the verbatim spoken content.
- Voiceover: "...says in an off-screen voiceover: <d>[English] ...</d> while his lips remain completely closed."
- Dialogue crossing a cut: use <scenetrans> and state the audio continues across the cut. Truncated speech: use <cutoff>.
- If no dialogue is spoken in a shot, describe visual reactions and body language instead.
- On-screen text (signs, labels, subtitles, neon text) in English double quotes, preserved verbatim: `A red neon sign reading "营业中" glows above the doorway.`

## 5. Audio & Music

End the response with exactly two lines:
Audio: [1-3 sentences of natural ambience, physical action sounds, footsteps, engine roar, breathing, chuckles — do NOT repeat the speech text here.]
Music: [1-2 sentences of non-diegetic score: instrumentation, tempo/BPM, dynamic rise or resolve across shots.]

## 6. Strict Output Rules

- Output ONLY the timed [Shot N] narrative paragraphs followed by the Audio: and Music: lines. NEVER write section headers, field names, or headings of any kind. NEVER add preamble such as "Based on the provided images" or "here is the description".
- NEVER invent custom tags like <Environment 1> or <Location 1> — reference assets only as <Subject N>, <Picture N>, <Video N>, <Audio N>.
- Write descriptions in English; dialogue, lyrics, and visible scene text remain in their ORIGINAL language.
- Total output should not exceed 7000 characters."""

# 每种生成模式的任务模板 (任务类型 + 对齐方式 + 输出结构)
H3_TASK_TEMPLATES = {
    "Text to Video (T2VA)": """## Task Type: Text-to-Video (T2VA)

This is a Text-to-Video generation task building the complete audiovisual timeline from text. No reference materials — no alignment instruction. Begin directly with [Shot 1] and cover the full timeline, then end with the Audio: and Music: lines.""",

    "Image to Video (I2VA)": """## Task Type: Image-to-Video (I2VA)

This is an Image-to-Video generation task. <Picture 1> is the starting frame at 0.00 seconds.

### Alignment Instruction (first line of the output, followed by one blank line):
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

### Output Structure:
[Shot 1] (Begins exactly matching the composition, lighting, and pose of <Picture 1>, then narrates the continuous forward motion)
[Shot 2] At MM:SS.mmm, ...
...
Audio: ...
Music: ...""",

    "First+Last Frame (FL2VA)": """## Task Type: First/Last Frame Timeline (FL2VA)

This is a First-and-Last Frame generation task. <Picture 1> is the starting frame (0.00s) and <Picture 2> is the ending frame.

### Alignment Instruction (first line of the output, followed by one blank line):
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

### Output Structure:
[Shot 1] (Narrates the continuous physical motion and transformation bridging the two reference frames)
...
Audio: ...
Music: ...

IMPORTANT bracket rule: in FL2VA ONLY, write Picture 1 / Picture 2 WITHOUT angle brackets and Shot 1 / Shot N WITHOUT square brackets — both in the alignment instruction and in the body. Favor a single continuous shot so the model can interpolate between the frames.""",

    "Last Frame (L2VA)": """## Task Type: Last Frame to Video (L2VA)

This is a Last-Frame generation task. <Picture 1> is the ending frame at S.SS seconds.

### Alignment Instruction (first line of the output, followed by one blank line):
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.

### Output Structure:
[Shot 1] ... (Infers a plausible preceding state and narrates the causal action converging on the final frame)
...
Audio: ...
Music: ...""",

    "Full Reference (Ref2VA)": """## Task Type: Reference Video/Image to Video-Audio (Ref2VA / Omni Reference)

This task uses reference media to transfer subject identities, characters, creatures, or motion into the target video.

### Directing & Staging Instructions:
1. Treat all available reference assets (<Subject N>, <Picture N>, <Video N>, <Audio N>) as members of a cohesive scene cast. <Subject N> identifies reusable visible content (person, scene, clothing, style, action); when a referenced subject speaks, write <Subject N> (Sx) where (Sx) identifies the actual speaker.
2. For multiple references in short durations (4-6s), use spatial staging (foreground/midground/background) to showcase characters, creatures, and vehicles interacting in the same space rather than rapid separate cuts.
3. Obey the shot budget by duration: 4-6s: 1-2 shots; 7-10s: 2-3 shots; 11-15s: 3-5 shots.
4. Establish the overall style in 1-2 sentences BEFORE [Shot 1]. Insert each reference label at its first appearance and where its role applies; keep label meanings consistent throughout.
5. REPLACEMENT, NOT DESCRIPTION: when a source video is provided together with reference assets, the target video keeps the source's timeline, motion, camera movement, rhythm, and scene structure, but its subject(s), characters, or background are REPLACED by the reference assets. Do NOT restate what the source video looks like — write what the NEW target video must look like after the replacement.
6. Format the output strictly as timed [Shot N] paragraphs, followed by the Audio: and Music: lines. Do NOT add section headers or field names. Start with 'Video' / 'Cinematic style' style statement, then [Shot 1] — never with a heading like '##Prompt##'. Do not output a description of the source video.""",

    "Video to Video (V2V)": """## Task Type: Video-to-Video (V2V) - Replacement V2V

subject_definitions

- <Picture 1>: the target Picture 1. Use the complete appearance of <CHARACTER APPEARANCE DESCRIPTION: hair style and color, face shape, facial features, makeup, skin tone, body frame>. She/He wears <FULL OUTFIT DESCRIPTION: top, bottom, material, colors, detailed accessories>. The reference includes full front, side, and back views, plus close-up details of the facial features, fabric texture, and outfit draping.
- <Picture 2>: the additional performer Picture 2. Use the complete appearance of <CHARACTER APPEARANCE DESCRIPTION: hair style and color, face shape, facial features, makeup, skin tone, body frame>. She/He wears <FULL OUTFIT DESCRIPTION: top, bottom, material, colors, detailed accessories>. The reference includes full front, side, and back views, plus close-up details of the facial features, fabric texture, and outfit draping.
- <Video 1>: the Video 1. Preserve the original performer's <MOTION AND PERFORMANCE DESCRIPTION: choreography, body rhythm, limb movements, gestures, interactions>, , , , shot framing, timing, and embedded audio. The original environment serves only as the motion and performance reference and is fully replaced by in the output.
- <Picture 3>: the Picture 3 replacing the original environment — <Picture 3 DESCRIPTION: scene type, dominant colors, materials, light direction and color temperature, atmospheric elements, and constraints such as no text/logos>.

summary
video editing + reference generation. Replace the entire performer in <Video 1> with the complete character from <Picture 1>, including face, hair, skin tone, body proportions, and full outfit. Simultaneously introduce the additional performer from <Picture 2> as a second character who appears naturally alongside <Picture 1> in the same frame, matching the original performance rhythm, interaction timing, camera path, and framing. Replace the original background with <Picture 3>, recomposing lighting and reflections to fit the new environment for both characters. Keep every other element of the Video 1 unchanged: <ELEMENTS TO PRESERVE: choreography, foreground interactions, props, camera work, audio>. The transferred characters must remain temporally consistent and physically coherent throughout the entire sequence, and must sit naturally inside the Picture 3.

retention_analysis
- fully_preserved: <ELEMENTS TO PRESERVE IN FULL: choreography, body rhythm, limb movements, gestures, interaction timing, prop position, camera path, framing, overall handheld feel, original audio track>.
- background_replaced: the entire original environment of <Video 1> is fully replaced by <Picture 3>. The new background's perspective, reflections, and lighting must be recomposed to match the original camera angle and the two transferred characters, with no mismatched shadows, reflections, or color casts.
- attribute_transfer: transfer the complete physical identity from <Picture 1> onto the main performing character: , and the complete outfit: . Transfer the complete physical identity from <Picture 2> onto the additional performing character: , and the complete outfit: .
- partially_preserved: preserve the original motion timing, body-driven movement dynamics, and choreography flow, but replace the performer's appearance, silhouette, skin, hair, face, and clothing with the character from <Picture 1>. The additional character from <Picture 2> must adapt to the same timing and interaction dynamics while preserving her/his own appearance, silhouette, skin, hair, face, and clothing. Both outfits must move naturally with the performance physics, responding to body movements in a physically plausible way.

background_replacement
Replace the entire original environment of <Video 1> with <Picture 3>. Keep the camera path, framing, and identical — only the environment behind and around the performers changes. Keep unchanged in appearance and timing. Re-light the scene to the new background's mood while preserving a key light on both performers so their faces, skin, and outfits remain consistent with <Picture 1> and <Picture 2> throughout the clip. Floors and reflective surfaces must show believable reflections of both performers consistent with their new outfits and the camera angle. The background must contain no text, logos, or readable signage unless specified, and must not introduce new foreground objects that occlude the performers or the interacting elements.

detailed_description
[Shot 1] Use <Video 1> as the exact motion, performance, camera, and timing reference for one continuous shot, and <Picture 3> as the new environment. Replace the entire performer with the complete character from <Picture 1>, wearing , while performing the same in <Picture 3> — . Introduce the additional performer from <Picture 2>, wearing , as a second character who performs alongside <Picture 1> in the same frame, following the original choreography, interaction timing, and camera movement. The main performer's flows with the momentum, maintaining its texture; the facial features and skin tone replace the original entirely and stay consistent across all visible body parts. The additional performer's flows with the momentum, maintaining its texture; the facial features and skin tone stay consistent across all visible body parts. The performers interact with , matching the original timing precisely, and continue to hold throughout the sequence. The camera maintains the same feel with in the lower-right foreground, now set against the new background. The transferred characters must maintain strict temporal consistency: no face drifting, no hair color shifting, no skin tone fluctuation, no body proportion changes, and no outfit texture or color inconsistencies across the entire clip. Lighting and reflections must stay stable and coherent with the new background. must deform naturally with each movement, staying physically attached without clipping or floating; must move believably with the body motions.

overall_soundscape
Preserve the original audio from <Video 1> entirely, including <AUDIO COMPONENTS: background music, rhythmic beats, vocals>. Adjust only the ambient room tone to match the new background: replace the original acoustics with <Picture 3 ACOUSTICS DESCRIPTION>.

non_diegetic_music
<AUDIO STRATEGY OPTION: Retain the original music track and its exact timing from <Video 1> / Replace with  with timing synced to the performance.>
""",
}


def build_h3_system_prompt(generation_mode, output_language="仅英文"):
    """组装 H3 系统提示词: 导演底座 + 所选生成模式的任务模板。

    generation_mode 可传完整下拉标签 ('H3 - 全参考 (Ref2VA)') 或英文核心名;
    output_language 仅英文 = 直接英文输出; 仅中文 = 附加强制中文指令
    (中文版通常由独立翻译 pass 生成, 此参数备用)。
    """
    if generation_mode:
        mode = normalize_h3_mode(generation_mode)
    else:
        mode = "Text to Video (T2VA)"
    task = H3_TASK_TEMPLATES.get(mode, H3_TASK_TEMPLATES["Text to Video (T2VA)"])
    lang_rule = ""
    if output_language == "仅中文":
        lang_rule = ("\n\n## 0. Language\nCRITICAL: write the ENTIRE output in Simplified Chinese "
                     "(简体中文). Dialogue and on-screen text remain in their original language.")
    return H3_SYSTEM_PROMPT_BASE + lang_rule + "\n\n" + task


# ============================================================
# Qwen-Image-2.1 system prompts
# ============================================================

QWEN_IMAGE21_SYSTEM_T2I = """You are a Qwen-Image-2.1 prompt rewriting expert. Rewrite the user's request as one polished, richly detailed English image-generation prompt. When the request is brief, expand it into a fully visualized, coherent still image rather than merely paraphrasing it. Aim for roughly 120-220 words when the concept supports that level of detail; use fewer words only when the user explicitly asks for brevity or the idea is inherently minimal. Add concrete, mutually consistent visual information across these dimensions: image medium and style; main subject's appearance, pose, expression, clothing or surface features; setting and specific environmental details; foreground, middle ground, and background; subject placement, scale, gaze, contact, and interaction with nearby elements; composition, framing, viewpoint, lens or depth-of-field character where useful; direction, softness, colour, and falloff of light; cast shadows, reflected light, diffuse light, material response, and reflections; palette, texture, atmosphere, and small details that strengthen the scene. Describe observable visual facts, not abstract praise or a list of disconnected keywords. Make the subject feel physically present in its environment, with plausible depth, contact, shadows, ambient illumination, and material reflectance. Preserve the user's core idea, requested style, mood, named subjects, and every explicit constraint. Elaborate only with details that naturally support that intent; do not invent story-critical facts, extra prominent subjects, logos, or text. Preserve explicit constraints exactly, especially quoted on-image text, spelling, capitalization, line breaks, quantity, colour, and position. State on-image text in double quotes and identify its carrier, placement, typography, colour, and visual treatment. Do not add negative prompts, model parameters, headings, explanations, or commentary. Output only the final prompt."""

QWEN_IMAGE21_SYSTEM_EDIT = """You are a Qwen-Image-2.1 image-editing prompt rewriting expert. Rewrite the user's request as one direct, precise English edit instruction. Identify the requested change and make it unambiguous; explicitly preserve all unrelated subjects, composition, identity, pose, lighting, scene, and visible text. When describing a person or object, state its natural relationship with the surrounding environment where relevant: placement, contact or interaction, depth, and physically coherent ambient light, cast shadows, reflected light, diffuse light, and material reflectance. For multiple reference images, use the exact tags <image1>, <image2>, etc. to name their roles; for one image, refer to it naturally as the image. Preserve every explicitly requested text string exactly, including spelling, capitalization, punctuation, line breaks, placement, and typography. Do not add negative prompts, parameters, headings, explanations, or commentary. Output only the final prompt."""

PROMPT_TRANSLATE_TO_CHINESE_SYSTEM = """Translate the supplied optimized generation prompt from English into Chinese.

Preserve every requirement, field name, reference label, tag, timestamp, number, quoted on-image text, dialogue, lyric, and formatting structure exactly. Translate descriptive prose only; dialogue, lyrics, and visible text must remain in their original language. Do not optimize, rewrite, add, omit, summarize, explain, or add headings. Output only the translated Chinese prompt."""


# 参考图自动反推提示词 (用于优化前对每张参考图生成内容描述)
IMAGE_CAPTION_PROMPT = (
    "Describe this image in detail, including the main subject, appearance, clothing, pose, "
    "expression, setting, colors, lighting, and composition. Output the description only, "
    "without any preamble."
)


VIDEO_CAPTION_PROMPT = (
    "The attached images are frames evenly sampled from ONE video, in chronological order. "
    "Describe this video as a whole for a text-to-video prompt: the main subject(s) and their appearance, "
    "the actions and motion evolving over time, camera movement (if any), scene and background, lighting and style. "
    "Use phrases like 'at first / then / finally' for temporal changes. Output the description only, "
    "without any preamble."
)


def build_video_reference_block(video_description, replacement=False):
    """将源视频整体反推描述格式化为注入优化请求的文本块。

    replacement=True: 同时提供了参考素材 (人物/背景), 此时源视频只作为「结构模板」——
    保留时间线/运镜/节奏, 但人物与背景必须被参考素材替换, 禁止复述原视频描述。
    """
    desc = (video_description or "").strip()
    if not desc:
        return ""
    lines = [
        "[Source Video] The caption below is an auto-description of the SOURCE video. The source video "
        "frames themselves are NOT attached to this request - only the user's reference assets are."
    ]
    if replacement:
        lines.append(
            "[Replacement Rule] Use the caption ONLY as a structural template: keep its timeline, motion, "
            "camera movement, rhythm, shot order and scene layout. REPLACE every subject, character, "
            "costume, prop and background with the reference assets described in "
            "[Reference Image Captions]. Do NOT reproduce, paraphrase or summarise that caption. "
            "The output must read as the prompt of a NEW target video, never as a description of the "
            "source video."
        )
    else:
        lines.append(
            "[Usage] Follow the source video's camera movement, rhythm and shot structure while applying "
            "the requested visual changes; rewrite it as the TARGET video's prompt, not as a caption."
        )
    lines.append("[Source Video Caption] " + desc)
    return "\n".join(lines)


def build_reference_captions_block(image_descriptions, start_index=1):
    """将参考图反推描述格式化为注入优化请求的文本块。

    start_index: 第一张图片的参考标签序号 (混合输入时视频帧排在最前, 图片标签从其后开始),
    使 "Image N" 与实际标签 <imageN>/<Picture N> 对齐。
    """
    if not image_descriptions:
        return ""
    lines = [
        "[Reference Image Captions] Auto-caption of each reference image "
        "(Image N corresponds to the reference label <imageN>/<Picture N>):"
    ]
    for i, cap in enumerate(image_descriptions):
        cap = (cap or "").strip()
        if cap:
            lines.append(f"Image {start_index + i}: {cap}")
    return "\n".join(lines) if len(lines) > 1 else ""


# 用户文字意图优先级规则: 自定义提示词中描述的动作/事件(如唱歌、跳舞、对话)即使
# 参考图中不可见甚至与画面矛盾, 也必须保留在优化结果中; 参考图只提供外观/场景/风格。
USER_INTENT_PRIORITY_RULE = (
    "[Priority Rule] The User Original Prompt is the PRIMARY creative intent and has the HIGHEST priority. "
    "Every action, event, behavior, and requirement it describes (e.g. singing, dancing, speaking, crying) "
    "MUST be explicitly and faithfully preserved in the optimized prompt, even if the reference images do not "
    "show that action or seem to suggest something different (e.g. the subject holds an instrument - if the user "
    "says she is singing, she IS singing). Reference images ONLY provide subject appearance, clothing, props, "
    "scene, and style; they must NEVER override, replace, or omit the actions and content requested by the user."
)

# 输出格式优先级: 输出结构只由所选生成模式的 skill 规范决定; 用户提示词只是内容引导,
# 即使其中包含自己的格式要求 (段落/标题/时间戳样式等), 也绝不能替换 skill 要求的输出结构,
# 用户的内容要求必须合并进 skill 结构的对应位置中。
FORMAT_PRIORITY_RULE = (
    "[Output Format Rule] The output structure is dictated SOLELY by the skill specification of the "
    "selected generation mode (mode hint above). The User Original Prompt is a CONTENT guidance only - "
    "NEVER follow its phrasing, headings, sections, or format. Even if the user prompt contains its own "
    "formatting instructions (e.g. shot lists, timestamp layouts, section titles, keyword lists), you MUST "
    "NOT copy them as the output skeleton. Instead, reorganize ALL user content requirements (camera work, "
    "color tone, lighting, shot scale, duration, style, etc.) INTO the exact output structure required by "
    "the selected mode's skill (for H3 modes: timed [Shot N] paragraphs ending with 'Audio:' and 'Music:' "
    "lines; for Qwen-Image modes: the direct instruction/description), then output that structure directly "
    "without any preamble such as 'Based on the provided images' or 'here is the description'."
)


# 主推理兜底指令: 当提示词优化开启时, 主推理拿到的是「优化后的提示词」而非聊天问题,
# 若不显式约束, VLM 会直接把附带的原视频/图片再描述一遍, 导致 output 退化成素材描述。
MAIN_INFERENCE_PROMPT_RULE = (
    "[Output Rule] The text above is the FINAL generation prompt you must produce. "
    "Rewrite and output that prompt directly for the target video/image, following the attached reference "
    "media as the source of subject identity, appearance and background. "
    "Do NOT describe the attached media, do NOT answer as a chatbot, do NOT add explanations or headings. "
    "Output only the final prompt text."
)


def build_qwen_image_system_prompt(generation_mode, output_language):
    """Build the Qwen-Image-2.1 monolingual system prompt; call twice for bilingual."""
    base = (
        QWEN_IMAGE21_SYSTEM_EDIT
        if "图像编辑" in generation_mode or "多参考图" in generation_mode
        else QWEN_IMAGE21_SYSTEM_T2I
    )
    if output_language != "仅中文":
        return base
    return (
        base.replace("English image-generation prompt", "Chinese image-generation prompt")
        .replace("English edit instruction", "Chinese edit instruction")
        + " Write all descriptive prose in Chinese; preserve explicitly requested on-image text exactly."
    )


# ============================================================
# Generation modes
# ============================================================

H3_MODES = (
    "H3 - 文生视频 (T2VA)",
    "H3 - 图生视频 (I2VA)",
    "H3 - 首尾帧 (FL2VA)",
    "H3 - 尾帧 (L2VA)",
    "H3 - 全参考 (Ref2VA)",
    "H3 - 替换v2v",
)

QWEN_IMAGE_MODES = (
    "Qwen-Image-2.1 - 文生图",
    "Qwen-Image-2.1 - 图像编辑",
    "Qwen-Image-2.1 - 多参考图",
)

# Single flat combo list: the family is encoded in the mode name itself,
# so no frontend JS is needed to filter values.
PROMPT_OPTIMIZER_MODES = list(H3_MODES) + list(QWEN_IMAGE_MODES)

# 中文下拉名 → H3 系统提示词内部使用的英文模式名
H3_MODE_ZH_TO_EN = {
    "文生视频 (T2VA)": "Text to Video (T2VA)",
    "图生视频 (I2VA)": "Image to Video (I2VA)",
    "首尾帧 (FL2VA)": "First+Last Frame (FL2VA)",
    "尾帧 (L2VA)": "Last Frame (L2VA)",
    "全参考 (Ref2VA)": "Full Reference (Ref2VA)",
    "替换v2v": "Video to Video (V2V)",
    "自动": "Auto",
}


def normalize_h3_mode(generation_mode):
    """'H3 - 文生视频 (T2VA)' -> 'Text to Video (T2VA)'"""
    core = generation_mode.strip()
    if core.startswith("H3 - "):
        core = core[len("H3 - "):]
    return H3_MODE_ZH_TO_EN.get(core, core)


# ============================================================
# User-message builders
# ============================================================

H3_MODE_HINTS = {
    "Text to Video (T2VA)": "Current mode: T2VA (Text to Video). No reference materials — build the complete audiovisual timeline from text. No alignment instruction; begin directly with [Shot 1], end with the Audio: and Music: lines.",
    "Image to Video (I2VA)": "Current mode: I2VA (Image to Video). <Picture 1> is the starting frame at 0.00 seconds. Use the first-frame alignment instruction, then narrate continuous forward motion from the reference frame.",
    "First+Last Frame (FL2VA)": "Current mode: FL2VA (First+Last Frame). <Picture 1> is the starting frame (0.00s), <Picture 2> the ending frame. Bridge them with continuous physical motion. IMPORTANT: in FL2VA only, write Picture N WITHOUT angle brackets and Shot N WITHOUT square brackets, both in the alignment instruction and the body.",
    "Last Frame (L2VA)": "Current mode: L2VA (Last Frame). <Picture 1> is the ending frame at S.SS seconds. Infer a plausible preceding state and narrate the causal action converging on the final frame. Alignment instruction: 'How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.'",
    "Full Reference (Ref2VA)": "Current mode: Ref2VA (Full Reference). Reference media (character images, action videos, scene images, music) transfer identities and motion into the target video. Define <Subject N>/<Picture N>/<Video N>/<Audio N> labels at first appearance, establish style in 1-2 sentences before [Shot 1], stage multiple references together in the same frame, and output timed [Shot N] paragraphs + Audio:/Music: lines (NO section headers).",
    "Video to Video (V2V)": "Current mode: V2V (Video to Video). The attached reference labels are frames sampled from ONE source video <Video 1>. Follow its camera movement, rhythm, and shot structure while applying the requested visual changes; output timed [Shot N] paragraphs + Audio:/Music: lines (NO section headers).",
}


# ============================================================
# 风格技能 (源自 MiniMax 官方 H3 skills 包的 8 个风格生成器)
# 官方技能是 MiniMax Hub agent 工作流技能; 此处提炼其风格要义,
# 作为提示词优化时的风格指令注入, 输出仍遵循 h3-prompt-writing 格式。
# ============================================================

STYLE_SKILLS = {
    "3D动画短片": (
        "Style: complete stylized 3D animated short. Pixar-inspired 3D cartoon rendering "
        "(C4D + Octane look, high-end animated-feature quality); exaggerated geometric character "
        "simplification with rich material detail, ~2.5-3 head-tall friendly proportions, big heads, "
        "compact bodies, highly readable silhouettes. Story-first structure with strict character "
        "consistency and scene continuity across all shots; control timing, camera, performance, and audio."
    ),
    "品牌宣传视频": (
        "Style: brand promo short. Highlight product capabilities, key use cases, and a clear call to "
        "action; precise beat/shot planning; describe only verifiable brand assets supplied by the user; "
        "clean premium camera language; suited for launches, website showcases, and social promotion."
    ),
    "合作游戏开场": (
        "Style: two-player co-op game menu / opening animation. Feature two characters with player cards "
        "and menu interaction motion; lock identity cues across shots; coordinated color scheme, buttons, "
        "icons, and typography; energetic game-trailer pacing."
    ),
    "手绘实拍融合": (
        "Style: surreal short blending rough glowing hand-drawn animation with live-action space. "
        "Crayon/chalk-like glowing stroke texture; a drawn object makes physical contact with the real "
        "world, then continuous morphing and a delayed handheld chase camera; playful tone, never horror."
    ),
    "极简产品广告": (
        "Style: clean minimalist product ad short. Distill selling points into concise ad copy; build "
        "stable product anchors; beat-synced typography and storyboards; premium camera language for "
        "e-commerce promotion and product launches."
    ),
    "音乐视频字幕": (
        "Style: music video with beat-reactive lyric typography. Analyze beat and vocal timing; keep "
        "character, scene, and on-screen text references separate; plan lyric text moments; decompose the "
        "song into connected shots with a consistent visual identity."
    ),
    "纸拼贴解说": (
        "Style: tactile paper-collage explainer. Visual metaphors built from cut-paper layers and halftone "
        "textures; stop-motion paper movement with tactile sound effects; expressive handmade look for "
        "knowledge points, opinions, and story visuals."
    ),
    "纸艺定格解说": (
        "Style: papercraft stop-motion explainer. Paper characters, layered diorama sets, and props in a "
        "cut-paper / pop-up-book / miniature look; explain the topic through staged reveals with character "
        "and scene continuity."
    ),
    "真实类型": (
        "Style: photorealistic live-action. Maintain photographic realism with natural skin texture, "
        "accurate lighting, real-world physics, and authentic material responses. Preserve fine facial details, "
        "natural color grading, and realistic depth of field. No cartoon stylization, no anime aesthetics. "
        "Emphasize real-world camera language: shallow depth of field, natural bokeh, realistic shadows and reflections."
    ),
    "动漫类型": (
        "Style: anime / manga-inspired animation. Clean line art, vibrant cel-shaded colors, stylized character "
        "designs with expressive eyes and simplified proportions. Dynamic camera moves typical of anime, "
        "exaggerated expressions and motion lines. Backgrounds with painterly or flat color fills. Maintain "
        "consistent anime aesthetic throughout all shots, with clear cel boundaries and iconic anime lighting."
    ),
}

# 自动推断关键词 (对用户提示词做匹配, 命中最多的技能生效; 无命中则不注入风格)
STYLE_SKILL_KEYWORDS = {
    "3D动画短片": ("3d", "三维", "动画短片", "皮克斯", "卡通渲染", "动画电影", "q版"),
    "品牌宣传视频": ("品牌", "宣传", "推广", "品牌片", "brand", "promo"),
    "合作游戏开场": ("游戏", "开场动画", "双人", "合作", "玩家", "game intro"),
    "手绘实拍融合": ("手绘", "实拍", "发光线条", "蜡笔", "粉笔", "融合"),
    "极简产品广告": ("产品", "广告", "电商", "极简", "新品", "带货", "product ad"),
    "音乐视频字幕": ("音乐视频", "mv", "歌词", "字幕", "卡点", "music video"),
    "纸拼贴解说": ("纸拼贴", "拼贴", "剪纸", "半调", "halftone"),
    "纸艺定格解说": ("纸艺", "定格", "立体书", "纸偶", "微缩", "papercraft", "stop motion"),
    "真实类型": ("真实", "照片", "写实", "photorealistic", "realistic", "real", "photo", "实拍", "写实风", "真人"),
    "动漫类型": ("动漫", "anime", "动漫风", "漫画", "cartoon", "动画", "二次元", "japanese anime", "manga"),
}


def resolve_style_skill(user_text):
    """自动模式: 按关键词命中数推断风格技能; 无命中返回 None (不注入风格)。"""
    text = (user_text or "").lower()
    if not text.strip():
        return None
    best, best_hits = None, 0
    for label, kws in STYLE_SKILL_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        if hits > best_hits:
            best, best_hits = label, hits
    return best


def build_style_skill_block(style_label):
    """将风格技能格式化为注入优化请求的文本块; 无效标签返回空串。"""
    directive = STYLE_SKILLS.get(style_label)
    if not directive:
        return ""
    return "[Style Skill] " + directive


def build_h3_user_message(user_prompt, generation_mode, image_count=0,
                          no_bgm=False, extra_requirements="",
                          image_descriptions=None, video_description=None,
                          caption_start_index=1, style_skill=None):
    """Build the H3 user message. generation_mode may be the full combo label
    ('H3 - 文生视频 (T2VA)') or the English core name. image_descriptions is the
    auto-caption list of reference images (index N -> <Picture N+1>);
    video_description is the whole-video caption when the references are video frames;
    caption_start_index is the reference label number of the first captioned image;
    style_skill is the selected/auto-resolved style skill label (Chinese)."""
    generation_mode = normalize_h3_mode(generation_mode)
    parts = [f"[Generation Mode] {generation_mode}"]

    if extra_requirements and extra_requirements.strip():
        parts.append(f"[Extra Requirements] {extra_requirements.strip()}")

    parts.append(f"[Mode Hint] {H3_MODE_HINTS.get(generation_mode, '')}")
    _style_block = build_style_skill_block(style_skill)
    if _style_block:
        parts.append(_style_block)
    prompt_text = (user_prompt or "").strip()
    if prompt_text:
        parts.append(f"[User Original Prompt]\n{prompt_text}")
        parts.append(USER_INTENT_PRIORITY_RULE)
        parts.append(FORMAT_PRIORITY_RULE)
    else:
        parts.append("[User Original Prompt]\nNo text prompt was provided. Follow the selected mode and use any attached reference images as context.")

    _has_ref_assets = bool(image_descriptions and any((c or "").strip() for c in image_descriptions))
    if image_count > 0 or video_description:
        _vid_block = build_video_reference_block(video_description, replacement=_has_ref_assets)
        _prefix = _vid_block + "\n" if _vid_block else ""
        if image_count > 0:
            parts.append(
                f"{_prefix}[Reference Assets] {image_count} reference image(s) are attached, labeled "
                f"<Picture 1>..<Picture {image_count}> in order. They are the ASSETS TO INJECT into the "
                "target video (subject identity, appearance, costume, background), NOT the source video.\n"
                "Please write clear labels and usage for each image. "
                "Analyze subject appearance, scene style, composition, etc. from the images and incorporate into the prompt optimization.\n"
                + (build_reference_captions_block(image_descriptions, caption_start_index) + "\n" if image_descriptions else "")
                + "If the user prompt refers to images by ordinal names such as \"图1/图2\", \"图片1\", or \"image 1\", "
                "map them in order to the standard labels <Picture 1>..<Picture N> and keep the mapping consistent."
            )
        elif _prefix:
            parts.append(_prefix.rstrip("\n"))
        if generation_mode == "Text to Video (T2VA)" and image_count > 0:
            parts.append(
                "[Note] Images detected. Please optimize using 'Image to Video' or 'Full Reference' mode."
            )
        if video_description and _has_ref_assets:
            # 源视频 + 参考素材同时存在: 明确「保留视频结构, 替换人物/背景」的重写目标,
            # 避免模型只把原视频描述一遍。(仅视频无参考素材时不注入, 避免凭空替换)
            parts.append(
                "[Rewrite Goal] This is a REPLACEMENT rewrite, not a description task. The source video is "
                "described in [Source Video Caption] and the new assets in [Reference Image Captions]. "
                "Step 1 - keep ONLY the source video's structure: timeline, motion, camera movement, rhythm, "
                "shot order and scene layout. Step 2 - REPLACE the original subject(s), character(s), costume "
                "and background with the reference assets. Step 3 - write the result as the TARGET video's "
                "prompt in this mode's format. The source video's identity must NOT be preserved unless the "
                "user explicitly asks for it. Never output a plain description of the source video."
            )

    if no_bgm:
        parts.append(
            "[Music] No background music is desired. End the prompt with 'Music: N/A'."
        )
    else:
        parts.append(
            "[Music] End the prompt with a 'Music:' line describing a non-diegetic score that matches the mood, style, and pacing of the video (instrumentation, tempo, dynamics). Do NOT use N/A unless absolutely no music fits the scene."
        )

    _goal = (
        "Write the TARGET video prompt AFTER the requested replacement, NOT a description of the source video. "
        if (video_description and _has_ref_assets)
        else "Write the TARGET video prompt, not a caption of the provided material. "
    )
    parts.append(
        "\nBased on the above information, optimize the prompt according to the H3 skill of the selected mode. "
        "The output structure MUST follow that mode's task template exactly: timed [Shot 1]...[Shot N] narrative "
        "paragraphs (with alignment instruction first where the task type requires it), ending with the 'Audio:' "
        "and 'Music:' lines. NO section headers, NO field names, NO user-prompt headings, no preamble. "
        + _goal + "Output the optimized prompt directly."
    )
    return "\n".join(parts)


def build_qwen_image_user_message(user_prompt, generation_mode, image_count=0,
                                  extra_requirements="", image_descriptions=None,
                                  video_description=None, caption_start_index=1):
    """Build the Qwen-Image-2.1 user message. image_descriptions is the auto-caption
    list of reference images (index N -> <image N+1>); video_description is the
    whole-video caption when the references are video frames; caption_start_index is
    the reference label number of the first captioned image."""
    mode_hint = {
        "Qwen-Image-2.1 - 文生图": "Task: text-to-image. Build a complete image description from text.",
        "Qwen-Image-2.1 - 图像编辑": "Task: image editing. Apply only the requested changes and preserve every unrelated visual attribute.",
        "Qwen-Image-2.1 - 多参考图": "Task: multi-reference image composition/editing. State the role and retained attributes of each supplied reference image using <image1>, <image2>, etc.",
    }.get(generation_mode, "Task: image generation.")
    parts = [mode_hint]

    prompt_text = user_prompt or ""
    has_nine_grid = "九宫格" in prompt_text
    has_storyboard = "故事板" in prompt_text or "分镜头" in prompt_text
    is_t2i = generation_mode == "Qwen-Image-2.1 - 文生图"
    is_multi_reference = generation_mode == "Qwen-Image-2.1 - 多参考图"

    if has_nine_grid:
        if is_t2i:
            grid_rule = (
                "[九宫格规则：已由用户明确触发] 按用户要求创作严格的三行三列九个画格，"
                "明确从左到右、从上到下的阅读顺序，保持九格边界、间距和整体版式清楚。"
                "逐格呈现用户要求的内容；若用户表达连续事件，保持人物、服装、道具、空间和动作状态连续。"
                "仅凭'九宫格'不添加镜号、时间码、对白、标题或制作注释；只有同时触发故事板规则或用户明确要求时才添加。"
            )
        elif is_multi_reference:
            grid_rule = (
                "[九宫格规则：已由用户明确触发] 创作严格的三行三列九个画格，明确阅读顺序并保持版面清楚。"
                "先逐张绑定参考图标签与用途，再说明每格实际使用哪些参考图；保持身份、服装、产品及场景来源对应稳定。"
                "若用户表达连续事件，保持跨格状态连续。仅凭'九宫格'不添加镜号、时间码、对白、标题或制作注释；"
                "只有同时触发故事板规则或用户明确要求时才添加。参考图标签是提示词引用，不是画面文字。"
            )
        else:
            grid_rule = (
                "[九宫格规则：已由用户明确触发] 若输入图本身是九宫格，按用户指定的画格或区域执行编辑，"
                "保持未指定画格、格线、间距、阅读顺序和整体布局不变；若用户要求重做整张版面，再按要求调整。"
                "若输入图不是九宫格而用户要求将其编排成九宫格，明确新布局及原图内容的保留方式。"
                "仅凭'九宫格'不添加镜号、时间码、对白、标题或制作注释；只有同时触发故事板规则或用户明确要求时才添加。"
            )
        parts.append(grid_rule)

    if has_storyboard:
        storyboard_rule = (
            "[故事板/分镜头规则：已由用户明确触发] 按制作型故事板组织画面；尊重用户指定的格数，"
            "未指定时根据剧情拍点决定，不因触发词自动强制九格。每格表现一个主要时刻，明确镜号或顺序、"
            "景别/视角、主体位置与动作，并保持人物、服装、道具、空间方向、光线和状态连续。"
            "默认在画格外设置清楚的注释区域；镜号、时间（仅在有时长依据时）、镜头说明、用户提供的对白/字幕、声音和转场"
            "均作为板面注释，不覆盖剧情画面。若用户明确要求纯视觉、无文字、无编号、不要说明或关键帧板，则省略注释与文字。"
            "逐字保留用户给定文字，不编造对白、品牌或剧情事实；无对白时不虚构台词。"
        )
        if is_multi_reference:
            storyboard_rule += (
                "先逐张说明每个 <imageN> 的身份、服装、产品、场景、姿态、风格或画布用途；"
                "再在每格标明实际使用的参考图标签，未出现在该格的素材不强行加入。"
            )
        elif not is_t2i:
            storyboard_rule += (
                "若编辑现有故事板，只修改用户指定的格子或板面元素，并保持其余画格及其注释对应关系不变。"
            )
        parts.append(storyboard_rule)

    if extra_requirements and extra_requirements.strip():
        parts.append(f"[Extra Requirements] {extra_requirements.strip()}")
    prompt_text = (user_prompt or "").strip()
    if prompt_text:
        parts.append(f"[User Original Prompt]\n{prompt_text}")
        parts.append(USER_INTENT_PRIORITY_RULE)
        parts.append(FORMAT_PRIORITY_RULE)
    else:
        parts.append("[User Original Prompt]\nNo text prompt was provided. Follow the selected prompt rules and use any attached reference images as context.")

    _has_ref_assets = bool(image_descriptions and any((c or "").strip() for c in image_descriptions))
    if image_count > 0:
        refs = ", ".join(f"<image{i + 1}>" for i in range(image_count))
        _vid_block = build_video_reference_block(video_description, replacement=_has_ref_assets)
        _prefix = _vid_block + "\n" if _vid_block else ""
        parts.append(
            f"{_prefix}[Reference Images] {image_count} image(s) uploaded: {refs}. "
            "Analyze their visible details before writing the optimized prompt.\n"
            + (build_reference_captions_block(image_descriptions, caption_start_index) + "\n" if image_descriptions else "")
            + "If the user prompt refers to images by ordinal names such as \"图1/图2\", \"图片1\", or \"image 1\", "
            "map them in order to the standard tags <image1>, <image2>, ... and use ONLY these standard tags in the output."
        )
    parts.append("Output the optimized prompt directly: the structure MUST be the direct edit instruction / image description required by this Qwen-Image-2.1 mode, NOT the user prompt's own headings or layout. No preamble or explanation.")
    return "\n".join(parts)


def build_messages(system_prompt, user_message, image_uris=None, image_label="<Picture {n}>"):
    """Build LLM messages, optionally multimodal.

    image_label is a template with a '{n}' placeholder, e.g. '<Picture {n}>' or '<image{n}>'.
    """
    if image_uris:
        user_content = [{"type": "text", "text": user_message}]
        for idx, uri in enumerate(image_uris):
            label = image_label.format(n=idx + 1)
            user_content.append({"type": "image_url", "image_url": {"url": uri}})
            user_content.append({"type": "text", "text": f"(Above is {label})"})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def split_bilingual_output(text):
    """Split an H3 bilingual reply into (english, chinese)."""
    text = text or ""
    cn_marker = re.search(r"(?is)<XZG_CHINESE_PROMPT>\s*", text)
    en_marker = re.search(r"(?is)<XZG_ENGLISH_PROMPT>\s*", text)
    if not cn_marker:
        cn_marker = re.search(
            r"(?im)^\s*[#>*`\-]*\s*(?:中文(?:版本|提示词)?|chinese\s+(?:version|prompt))\s*[:：\-]*\s*$",
            text,
        )
    if not en_marker:
        en_marker = re.search(
            r"(?im)^\s*[#>*`\-]*\s*(?:英文(?:版本|提示词)?|english\s+(?:version|prompt))\s*[:：\-]*\s*$",
            text,
        )
    if not cn_marker or not en_marker:
        # Model did not follow the format; keep the result to avoid silent loss.
        return text.strip(), ""

    if cn_marker.start() < en_marker.start():
        chinese = text[cn_marker.end():en_marker.start()].strip()
        english = text[en_marker.end():].strip()
    else:
        english = text[en_marker.end():cn_marker.start()].strip()
        chinese = text[cn_marker.end():].strip()
    return english, chinese
