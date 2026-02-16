# ColorQuest Animate — AI Coding Agent Instructions

## Project Overview

**ColorQuest Animate** is a three-step creative tool that turns user imagination into animated artwork:

1. **Generate**: Diffusion models create clean, thick-line black & white coloring pages from text prompts
2. **Color**: Users paint the generated page (paper, app, digital canvas)
3. **Animate**: Upload colored artwork → detect colored shapes → apply animation effects → export GIF

**Key constraint**: MVP must run locally, CPU-friendly, kid-friendly UI. No cloud required.

## Architecture Pattern

### Three Independent Pipelines (Work in Parallel)

```
User Input (Prompt)
       ↓
┌──────────────────────────────────────┐
│ 1. GENERATION PIPELINE               │
│ - Prompt validation & formatting     │
│ - Load diffusion model + ControlNet  │
│ - Generate clean line art (B&W only) │
│ - Return PIL Image (high contrast)   │
└──────────────────────────────────────┘

Colored Artwork (Image File)
       ↓
┌──────────────────────────────────────┐
│ 2. COLOR ANALYSIS PIPELINE           │
│ - Load user's colored version        │
│ - Segment regions by color           │
│ - Map colors to original line art    │
│ - Build color-to-shape map           │
└──────────────────────────────────────┘

Color Map + Original Image
       ↓
┌──────────────────────────────────────┐
│ 3. ANIMATION PIPELINE                │
│ - Apply effects (shake, pulse, move) │
│ - Frame generation per effect        │
│ - GIF encode + export                │
└──────────────────────────────────────┘
```

## Critical Implementation Conventions

### 1. Streamlit Structure (Frontend)

- **Single entry point**: `app.py` (no multi-page until v2)
- **Layout pattern**: Three tabs matching the workflow (Generate | Color | Animate)
- **State management**: Use `st.session_state` for inter-step data (generated image, upload, color map)
- **Model loading**: Cache expensive operations with `@st.cache_resource` (e.g. diffusion models, processors)
- **Error handling**: Use `st.error()` / `st.warning()` for user feedback; never fail silently

**Example tab structure**:
```python
# app.py
st.title("ColorQuest Animate")
tab1, tab2, tab3 = st.tabs(["Generate", "Color", "Animate"])

with tab1:
    # Prompt input → call generate_coloring_page()
    
with tab2:
    # Upload interface → display reference + store
    
with tab3:
    # Animation preview + export
```

### 2. Image Processing (OpenCV / Pillow)

- **Generation output**: Enforce 1024×1024, pure B&W (threshold at 128), no antialiasing
- **Color analysis**: Use HSV colorspace (more robust than RGB); avoid pure black/white regions
- **Animation frames**: Pre-generate all frames before GIF encoding; store as numpy arrays
- **GIF export**: Always use `duration=50` (20fps) for smooth kid-friendly playback
- **Format consistency**: RGB for all PIL Images; convert internally from BGR if using OpenCV

### 3. Model Loading & Inference

- **Diffusion pipeline**: Load via `from diffusers import StableDiffusionControlNetPipeline`
- **Base model**: `runwayml/stable-diffusion-v1-5` (CPU-friendly, well-documented)
- **ControlNet**: Use `lllyasviel/sd-controlnet-canny` for thick-line guidance (proven for children's book style)
- **Prompt templates**: Always wrap user prompts in context: `f"{user_prompt}, thick outlines, bold lines, children's coloring book style, no shading, no gradients"`
- **Safety filter**: Disable if generating takes >5s; log performance for v2 optimization
- **Device handling**: Auto-detect CPU vs GPU; fallback to CPU with clear user messaging if needed
- **Model caching**: Use `HF_HOME` environment variable to control download location; cache to `~/.cache/huggingface/` by default
- **Inference steps**: Start with 20 steps for drafts, 50 for final output; adjust based on performance testing

### 4. Module Organization (When Scaling Beyond app.py)

```
app.py                           # Streamlit UI entry point
modules/
  ├── generation.py             # Diffusion pipeline wrapper
  ├── color_analysis.py         # Segmentation & color detection
  ├── animation.py              # Effects + GIF encoding
  └── utils.py                  # Common helpers (validation, caching)
```

When refactoring: Keep each module **single-responsibility**; pass PIL Images / numpy arrays between, never file paths.

### 5. Data Flow Invariants

- **Generation input**: String prompt (validated, max 100 chars)
- **Generation output**: PIL.Image (mode="L" or "RGB", 1024×1024, 0-255 range)
- **Uploaded colored version**: PIL.Image (mode="RGB", user's resolution, stored in session_state)
- **Color map**: Dict[color_tuple] → List[contour_indices] (enables animation targeting)
- **Animation output**: GIF bytes (downloadable, <2MB for kid-friendly apps)

### 6. Performance Targets & Optimization

**CPU-based targets** (measured on 4-core Intel / AMD, 8GB RAM):

| Operation | Target Time | Implementation Hints |
|-----------|------------|----------------------|
| Generation (20 steps) | <30s draft | Use inference scheduler `EulerDiscreteScheduler`; skip NSFW check; profile with `timeit` |
| Generation (50 steps) | <90s final | Cache pipeline after first load; reuse seed for reproducibility |
| Color analysis | <5s | Use OpenCV's `findContours` (faster than scikit-image); pre-blur with `GaussianBlur` |
| Animation (50 frames) | <10s | Pre-compute frame effects in NumPy; avoid PIL operations in loop |
| GIF encoding | <5s | Use `PIL.ImageSequence` with optimized palette (256 colors max) |

**Memory constraints**:
- Diffusion pipeline: ~2.5GB RAM (quantize to fp16 if needed: `torch.float16`)
- Full flow (gen + color + animate): Never exceed 6GB; unload pipeline after use with `del pipeline; torch.cuda.empty_cache()`

**Debugging performance**:
```python
import time
start = time.time()
# ... operation ...
print(f"Operation took {time.time() - start:.2f}s")
```

### 7. Asset Management & Test Fixtures

**Directory structure for assets**:
```
assets/
  ├── test_fixtures/
  │   ├── bw_samples/              # Reference B&W generated coloring pages
  │   │   ├── dinosaur_clean.png   # ~30KB, 1024×1024
  │   │   ├── castle_thick.png
  │   │   └── flower_bold.png
  │   ├── colored_samples/         # User-colored versions for color analysis testing
  │   │   ├── dinosaur_colored.jpg # ~50KB, various resolutions (test upscaling)
  │   │   └── castle_colored.jpg
  │   └── expected_outputs/        # Reference GIF animations for regression testing
  │       └── dinosaur_animation.gif
  └── demo/                        # Public gallery samples (for future web showcase)
      └── featured_animations/
```

**Fixture creation workflow**:
1. Generate sample B&W at 1024×1024 with `inference_steps=50`; save as PNG (lossless)
2. Manually color 2-3 samples in Krita/Procreate at different resolutions (tablet 512px, phone 1080px)
3. For each colored sample, verify color detection finds all regions; document expected color_map in `fixtures/README.md`
4. Store reference GIFs with comments: `# dinosaur_animation.gif - shake effect, 15 frames, 280KB`

**CI integration** (when repo grows):
- Run `pytest tests/test_generation.py --fixture-dir assets/test_fixtures/` to validate against known good outputs
- Track fixture sizes in `.gitignore_large`: Store >100KB samples in git-lfs if needed

- **Unit tests**: One test file per module (`test_generation.py`, `test_animation.py`)
- **Test data**: Keep small reference B&W images in `tests/fixtures/` (PNG, <50KB each)
- **Integration tests**: Simulate full 3-step flow locally (no API calls)
- **No e2e tests yet** (Streamlit slow to test; deferred to v2)

### 8. Error Scenarios & Recovery Strategies

**Generation pipeline errors**:

| Error | Cause | User Message | Recovery |
|-------|-------|--------------|----------|
| **"Too many objects"** | Prompt too complex, OOM | "Let's try a simpler request! Use 1-2 objects (e.g., 'blue dragon' instead of 'dragon with 10 friends')" | Detect word count >20; suggest simplification |
| **Thin/light lines** | Not using canny ControlNet | "The lines look faint. Try: 'bold thick outlines' in your prompt" | Log ControlNet weights; verify canny guidance scale >0.5 |
| **Takes >2 minutes** | Model loading first time | Show progress: "Loading AI artist… (first time only, ~30s)" | Cache pipeline in `st.session_state`; show spinner with ETA |
| **CUDA OOM** | GPU memory exhausted | "Using CPU instead (will be ~2x slower). You can try a smaller prompt!" | Catch `RuntimeError`; fall back to CPU; log incident for v2 optimization |

**Color analysis errors**:

| Error | Cause | User Message | Recovery |
|-------|-------|--------------|----------|
| **"I can't find colors in this part"** | Colored outside lines / too light | "Upload a clearer photo! Make sure colors are bold inside the lines." | Visualize detected regions with `st.image(color_map_viz)`; highlight missing areas |
| **Wrong color detected** | Similar HSV values, bleeding | "Let me try again with those colors…" | Expand hue tolerance slightly; ask user to recolor OR auto-adjust HSV threshold +10% |
| **No regions found** | Completely white upload / wrong file | "'Oops! That doesn't look like our coloring page. Upload a photo of your colored version." | Check if >90% pixels are near-white; validate image size vs. original |

**Animation pipeline errors**:

| Error | Cause | User Message | Recovery |
|-------|-------|--------------|----------|
| **GIF too large (>5MB)** | Too many frames / high resolution | "Making it smaller so you can share it…" | Auto-reduce frames by 50%; downscale to 720px max width |
| **GIF export fails** | Palette generation error | "Let me process this differently… (might take a few seconds)" | Fallback: Use fixed 256-color palette instead of optimized |
| **No animation applied** | Color region unmapped to original | Show stuck frame; suggest: "Try uploading a clearer photo to detect all the colors." | Visualize unmatched regions; link to FAQ on proper coloring technique |

**Implementation approach**:
```python
# Generation wrapper with recovery
@st.cache_resource
def load_diffusion_pipeline():
    try:
        return StableDiffusionControlNetPipeline.from_pretrained(...)
    except torch.cuda.OutOfMemoryError:
        st.warning("GPU memory full. Switching to CPU (will be slower).")
        torch.cuda.empty_cache()
        return pipeline.to("cpu")

# Color analysis with visualization debugging
def detect_colors(colored_image, threshold=10):
    results = find_regions_hsv(colored_image, threshold)
    if len(results) == 0:
        st.warning("No colored regions found. Try uploading a clearer photo!")
        # Show what the algorithm "sees"
        debug_viz = create_hsv_mask(colored_image)
        st.image(debug_viz, caption="Debug: What I detected")
    return results
```

## Testing & Validation Conventions

1. **Mixing RGB/BGR**: OpenCV is BGR; convert to RGB immediately on load
2. **Thick lines not working**: Ensure ControlNet uses **canny edge** guidance, not segmentation
3. **Colors bleeding across line art**: Filter out near-black regions when mapping colors to shapes
4. **GIF file size**: Reduce frames aggressively; consider palette optimization for <1MB exports
5. **Model OOM on CPU**: Never load full model + pipeline together; lazy-load with early caching

## Development Workflow

- **Local testing**: `streamlit run app.py` opens browser automatically
- **Dependencies**: Pin versions in `requirements.txt` (avoid floating; diffusers especially volatile)
- **Hardware**: Works on CPU; test on actual target device (Raspberry Pi testing deferred to v2)
- **No build step**: Pure Python; no compilation needed

## Key Design Decisions (Why This Way)

| Decision | Rationale |
|----------|-----------|
| **Streamlit** | Fast iteration, no backend needed; drag-and-drop UI perfect for kids |
| **Diffusers** | Open-source, CPU-friendly, ControlNet support for thick lines |
| **Local-first** | Privacy, offline mode, classroom-friendly; cloud is v2 optional |
| **GIF export** | Universal format, embeddable, no external dependencies |
| **B&W generation** | Clearer for coloring; less ambiguous color regions to detect |

## Next Steps for AI Agents

When implementing features:
1. **Start with one pipeline** (usually Generation) — ensure clean line art first
2. **Add Color Analysis** — test with hand-drawn samples from real users (not synthetic)
3. **Build Animation** — start with simple effects (shake, color-pulse), add bouncing/walking later
4. **Polish UI** — big buttons, emojis, undo/retry flows (important for kids!)
5. **Optimize performance** — measure on CPU; profile model inference time

For debugging:
- Use `st.image()` to visualize intermediate outputs (B&W generated, color map, frames)
- Check logs for model loading issues; device detection errors are silent failures
- Save GIF test artifacts for code review

