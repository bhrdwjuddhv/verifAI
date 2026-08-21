# VerifAI Browser Extension — Implementation Plan

Revised against the phase-by-phase review. Every item below is anchored to code that already
exists in this repo, so the extension inherits the app's rule: **no verdict is better than a
guessed one.**

---

## Status (updated 2026-08-20)

| Phase | State |
|---|---|
| 0 — Scaffold, MV3 manifest, two-format Vite build | **done** |
| 1 — Right-click verify, `blob:` reads, cache, queue, consent | **done** |
| 2 — On-device inference in an offscreen document | **done, face-only** — NPR still needs an export (P1) |
| 3 — Occlusion saliency on-device | **done** — behind an explicit "Explain this score" action |
| Parity harness (Phase 2 ship gate) | **built and running** — blocked green by the missing NPR export |
| 4 — Feeds, Shorts, Reels | **built** — 11/11 in a synthetic feed harness; the 50-Shorts ship gate needs the real sites |
| 5 — Options, Firefox target, store packaging | **done** — both packages build; store submission is yours to run |

**Spikes S1 and S2 are answered** (measured in a Chromium tab via `scripts/preview.mjs`):

- **S1 — threads: no.** `crossOriginIsolated` is false and `SharedArrayBuffer` is undefined,
  so ORT runs single-threaded. The review was right; the manifest COEP/COOP keys do not
  rescue it.
- **S2 — WebGPU: yes.** An adapter is available and ORT selects the WebGPU EP. This is the
  one that mattered — it is why Phase 3's 26 forward passes stay interactive despite S1.
- **S2b — the CSP, found only in a real install.** MV3's default `script-src 'self'` forbids
  `WebAssembly.instantiate`, so every on-device scan failed with a `CompileError` until the
  manifest declared `'wasm-unsafe-eval'`. The browser-tab harness could not have caught it —
  it served pages over http with no extension CSP. It now serves the manifest's own policy,
  and reproduces the failure exactly when the token is missing.
- Measured on a 512×384 PNG with no face present: **3.7s cold** (21MB wasm + 16MB session),
  **1.0–1.4s warm**. Cold cost is dominated by loading, not by inference.

---

## 0. What the review changed

| Review finding | Resolution in this plan |
|---|---|
| Service worker can't run WebGPU / dies at ~30s | All inference moves to an **offscreen document** (`chrome.offscreen`, reason `WORKERS`). The SW only routes messages. → Phase 2 |
| MV3 blocks remote code | ORT `.wasm` + `.onnx` are **bundled**, `ort.env.wasm.wasmPaths` points at `chrome.runtime.getURL('ort/')`. No CDN anywhere. → Phase 2 |
| `captureVisibleTab` needs more than `tabs` | Frame grab is **canvas-first** (works for MSE/`blob:` video), `captureVisibleTab` is the fallback and needs `activeTab` (gesture) or host permission (auto mode). → Phase 4 |
| Grad-CAM impossible on-device | Correct — and already solved server-side: the torch-free path in `scripts/inference_server.py:288` uses **occlusion saliency**, not Grad-CAM. We port that exact function. It is the same feature, not a downgrade. → Phase 3 |
| No SharedArrayBuffer → single-threaded WASM | Treated as **an open question with a 30-minute spike** (S1 below), not an assumption. If threads are unavailable the heatmap becomes an explicit on-demand action with a progress bar. → Phase 2 |
| `blob:` URLs break server-mode scanning | Content script reads the blob and hands **base64** to the SW. Chrome's `runtime.sendMessage` is JSON-serialised — an `ArrayBuffer` does not survive the hop. → Phase 1 |
| Instagram/YouTube selectors are fragile | **No class-name selectors.** Generic `img`/`video` discovery + `MutationObserver` + per-element `WeakMap` state keyed off `loadstart`/`emptied` so a recycled player resets. → Phase 4 |

Two findings the review did not raise, which are larger risks than any of the above:

1. **Fusion parity.** The server fuses `face` and `npr` at 0.5/0.5 (`scripts/common/config.py`,
   `FUSION_DEFAULTS`). Shipping only the face model on-device produces a *different number* from
   the same image — the extension would contradict the website. **NPR must ship on-device too.**
   It is cheap: NPR is a truncated ResNet50 (trunk stops after `layer2`, one logit) ≈ 1.5M
   params, ~6MB fp32.
2. **Face-crop parity.** The face model has `expectsFace: true`. The server crops with
   `crop_face(margin=0.35)` — square, clamped, min half-side 24px. An uncropped or
   differently-cropped input silently shifts the score. The extension needs the same detector
   (YuNet ONNX, 340KB) and the same crop maths.

---

## 1. Ground truth from the repo (do not re-derive these)

| Thing | Value | Source |
|---|---|---|
| Face model | `models/face/detector.onnx`, 16.03MB, fp32 | committed to git |
| Preprocessing | 224px, ImageNet mean/std, labels `["Fake","Real"]`, `expectsFace: true` | `models/face/detector.json` |
| Calibration | temperature **0.7193467020988464**, applied to logits before softmax | same |
| Inference recipe | horizontal-flip TTA: run `[arr, flip(arr)]`, average the two probability rows | `inference_server.py:264` |
| Fake class index | matched **by label name**, never by index | `inference_server.py:69` |
| Face crop | YuNet score ≥ 0.7, margin 0.35, square, clamped, reject if half-side < 24px | `preprocess_faces.py:73` |
| NPR input | whole frame, **centre crop at native resolution** (`crop=True`), 224px, ImageNet norm, sigmoid on one logit | `inference_server.py:379`, `models/npr_model.py` |
| Fusion | `{face: 0.5, npr: 0.5, frequency: 0.0}`, renormalised over detectors that actually ran | `common/config.py` |
| Verdict bands | fake > 70, real < 30, else uncertain | `common/config.py` |
| Occlusion saliency | 5×5 grid (`VERIFAI_SALIENCY_GRID`), patches zero-filled *after* normalisation, drops clipped at 0, normalised by peak, `None` if peak ≤ 0 | `inference_server.py:288` |
| Overlay render | red where hot / blue where cold, `alpha=0.45`, weighted by the CAM itself | `common/xai.py`, `encode_overlay` |
| Public API | `POST /api/scan`, multipart field `file`, 50MB cap, `maxDuration = 60` | `app/api/scan/route.ts` |
| Rate limit | **20 requests / minute / IP** across all of `/api/*` | `middleware.ts` |
| Response shape | `ModelServiceResult` + app fields (`score`, `verdict`, `reasons`, `heatmap`, `fusion`, …) | `lib/models/model_service.ts` |

---

## 2. Prerequisites (do these before Phase 2 — half a day)

- [ ] **P1.** Export NPR to ONNX: `python scripts/export_onnx.py --npr`. Needs
      `models/npr_detector.pth` — either train it or pull the official checkpoint (the exporter
      prints the URL). Without this, on-device mode cannot match server mode.
- [ ] **P2.** Vendor YuNet: download `face_detection_yunet_2023mar.onnx` from opencv_zoo into
      `models/face_detection_yunet.onnx` (the server already looks there — `YUNET_PATH`).
      Un-ignore it in `.gitignore` alongside the face model.
- [ ] **P3.** Add `"exclude": ["node_modules", "extension"]` to the root `tsconfig.json`. Its
      `include` is `**/*.ts`, so extension sources with `chrome.*` types would otherwise break
      `next build`.
- [ ] **P4.** Record a golden fixture set: 30 images (10 real / 10 face-swap / 10 fully
      generated, at least 5 with no face) plus the exact JSON `/predict` returns for each. This
      is the parity oracle for the rest of the project.

---

## 3. Repo layout

```
extension/
  package.json              # own deps; NOT merged into the Next app's
  vite.config.ts            # multi-entry build + static copy of ort/ and models/
  manifest.chrome.json
  manifest.firefox.json     # background page instead of offscreen
  public/
    ort/                    # ort-wasm-simd*.wasm, *.jsep.wasm — copied at build
    models/                 # detector.onnx, detector.json, npr.onnx, yunet.onnx, models.json
  src/
    background/             # service worker: menus, routing, cache, queue, alarms
      index.ts  menu.ts  cache.ts  queue.ts  offscreen-host.ts
    offscreen/              # the only place inference happens
      index.html  index.ts  session.ts  preprocess.ts  face.ts  fuse.ts  saliency.ts
    content/                # DOM only: discovery, badges, overlay, blob reads, frame grab
      index.ts  discover.ts  badge.ts  overlay.ts  media.ts
    options/  popup/  onboarding/
    shared/
      scan-types.ts         # re-exports ../../lib/models/model_service types (one source of truth)
      protocol.ts           # every runtime message, discriminated union
      verdict.ts            # re-exports ../../lib/verdict VERDICT_CONFIG wording
  scripts/
    parity.mjs              # onnxruntime-node vs live FastAPI, over the P4 fixtures
    bench.mjs
```

`shared/scan-types.ts` importing across the folder boundary is deliberate: when the API response
shape changes, the extension fails to compile instead of silently mis-rendering.

---

## 4. Spikes to run first (1 day, before committing to Phase 2)

These three answers change the plan. Run them as throwaway code in a bare offscreen document.

- **S1 — threads.** Log `crossOriginIsolated` and `typeof SharedArrayBuffer` in the offscreen
  doc, with and without the `cross_origin_embedder_policy` / `cross_origin_opener_policy`
  manifest keys. The review says threads are impossible; verify rather than assume. **If threads
  work, every timing budget below improves ~3–4× and the heatmap becomes interactive.**
- **S2 — WebGPU.** `await navigator.gpu?.requestAdapter()` inside the offscreen doc, then a real
  ORT session on `detector.onnx` with `executionProviders: ['webgpu']`. Watch the console for
  per-op CPU fallbacks (EfficientNet's SiLU and depthwise convs are the ops to check). Record
  ms/forward for webgpu vs wasm.
- **S3 — canvas taint.** On youtube.com/shorts and instagram.com/reels, `drawImage(videoEl)` into
  a canvas and call `getImageData`. If it doesn't throw, `captureVisibleTab` is only a fallback
  and auto mode may not need broad host permissions at all.

**Gate:** if S2 fails *and* S1 fails, on-device inference is ~4–10s per heatmap and ~0.3–1s per
verdict. Still shippable, but the heatmap must then be an explicit button, never automatic.
Decide here, in writing, before building Phase 2.

---

## 5. Phases

### Phase 0 — Scaffold (1 day)

Vite multi-entry (plain Vite + `vite-plugin-static-copy`, not `@crxjs` — its MV3 plugin is still
beta and the HMR magic is not worth the churn risk). Entries: `background`, `offscreen`,
`content`, `options`, `popup`, `onboarding`. SW built as an ES module (`"type": "module"` in the
manifest's `background`).

Manifest baseline:

```jsonc
{
  "manifest_version": 3,
  "permissions": ["contextMenus", "storage", "offscreen", "scripting", "alarms"],
  "optional_host_permissions": ["https://*.youtube.com/*", "https://*.instagram.com/*"],
  "host_permissions": ["https://<verifai-host>/*"],
  "background": { "service_worker": "background.js", "type": "module" }
}
```

Site permissions are **optional** and requested from the options page. Asking for youtube.com and
instagram.com at install time is the fastest way to a slow Chrome Web Store review.

**Ship gate:** loads unpacked, SW logs, offscreen document opens and closes cleanly.

---

### Phase 1 — Right-click verify, server mode (2–3 days)

Flow: context menu on `image`/`video` → get `info.srcUrl` → fetch bytes → SHA-256 → cache hit? →
else queue → `POST /api/scan` → render result in the popup plus a badge on the element.

**The `blob:` fix.** `info.srcUrl` on WhatsApp Web and Instagram is `blob:https://…`, which the SW
cannot fetch. The context-menu click grants `activeTab`, so:

```
SW:   chrome.scripting.executeScript(tabId, func: readBlob, args: [srcUrl])
page: fetch(blobUrl) -> Blob -> FileReader.readAsDataURL -> base64 string
SW:   base64 -> Uint8Array -> Blob -> FormData -> POST
```

Base64 rather than `ArrayBuffer` because Chrome's `runtime.sendMessage` JSON-serialises (Firefox
structured-clones; base64 is correct on both). Budget the 33% overhead; cap at 10MB and say so
plainly above that.

**CORS.** No server change needed: fetches from the SW with a matching `host_permissions` entry
are exempt from CORS, custom headers included. Send `X-VerifAI-Client: extension/<version>` so
extension traffic is distinguishable in logs. Do **not** move the POST into the content script —
MV3 content-script fetches run under the *page's* CORS.

**Rate limit.** `middleware.ts` allows 20/min/IP for the whole public API, shared with anyone
browsing the site from the same IP. Queue at a **token bucket of 12/min**, serialise requests,
honour `Retry-After` on 429, and surface "queued — N ahead" rather than failing silently.

**Cache.** `chrome.storage.local`, key = SHA-256 of the bytes, value = the verdict JSON with
`heatmap` **stripped** (a 224px overlay PNG is 30–80KB; 10MB of quota disappears in ~150 scans).
Heatmaps go in a separate 3MB LRU. Entries carry the model version from `models.json` and are
invalidated when it changes.

**Consent.** First run opens `onboarding.html`: on-device vs deep scan explained in one screen
each, with the sentence "deep scan uploads the image to the VerifAI server" not buried. Store
`{consentVersion, acceptedAt}`; re-prompt when the version bumps. Deep scan is inert until
consent exists.

**Errors, honestly.** `/api/scan` distinguishes 501 / 415 / 413 / 503 and returns
`unavailable: true` for a down model service. Render each distinctly — never collapse them into
"scan failed", and never fall back to a local guess to paper over a server error.

**Ship gate:** right-click verify works on a normal `https://` image, on a `blob:` image on
WhatsApp Web, and on a cached repeat (instant, no network). Rate-limit path demonstrated by
firing 30 scans in a minute.

---

### Phase 2 — On-device mode (4–6 days)

**Offscreen host.** `ensureOffscreen()` in the SW: `chrome.offscreen.hasDocument()`, a
module-level in-flight promise to swallow the create race (only one offscreen document may
exist), `reason: ['WORKERS']`, justification "onnxruntime-web inference". The document is not
subject to the SW's idle timeout, but recreate defensively — the SW dying is normal, and the
offscreen document surviving that is not something to rely on.

**Sessions.** Three ORT sessions, created lazily and kept warm:

| Session | File | Size | Input |
|---|---|---|---|
| face | `detector.onnx` | 16MB fp32 | 1×3×224×224, ImageNet norm, from the face crop |
| npr | `npr.onnx` | ~6MB fp32 | 1×3×224×224, ImageNet norm, **centre crop at native res** |
| yunet | `yunet.onnx` | 340KB | dynamic, BGR uint8 |

**Ship fp32, not int8.** The committed `detector.onnx` is fp32, and the server itself notes that
int8 shifts scores "by a point or two". Quantising the extension copy would mean the extension
and the website disagree by design. 22MB of models in a CWS package is a one-time download;
parity is permanent. Revisit only if load time measures badly.

**ORT config:**

```ts
ort.env.wasm.wasmPaths = chrome.runtime.getURL('ort/');
ort.env.wasm.numThreads = crossOriginIsolated ? navigator.hardwareConcurrency : 1;  // see S1
const ep = (await navigator.gpu?.requestAdapter()) ? ['webgpu', 'wasm'] : ['wasm'];
```

**YuNet decoder.** ORT gives raw outputs; OpenCV's `FaceDetectorYN` post-processing has to be
hand-written: strides 8/16/32, per-stride cls/obj/bbox/kps heads, `score = sqrt(cls × obj)`,
threshold 0.7, NMS at 0.3, largest face wins. ~120 lines. Then `cropFace()` ports
`preprocess_faces.py:73` exactly — including the `half < 24 → null` rejection, which is the
difference between "no face" and a garbage 12px crop scored as evidence.

**Scoring pipeline** (mirror `analyze()` at `inference_server.py:490`):

1. YuNet → crop, or `faceDetected = false`.
2. Face session only if a face was found (`expectsFace` is true) — otherwise it **abstains**, it
   does not vote.
3. NPR on the whole frame, always.
4. Softmax with temperature 0.7193…, flip-TTA averaged, fake index by label name.
5. `fuse()` — weighted mean over detectors that ran, renormalised. Never let a missing detector
   drag the average toward 50.
6. `verdictFor()` — >70 fake, <30 real, else uncertain.
7. Emit the same `notes[]` the server emits, including "no face detected — the face classifier
   does not apply, so it did not vote".

**Parity harness (`scripts/parity.mjs`) — the single most valuable deliverable in this phase.**
Runs the P4 fixtures through onnxruntime-node against the same `.onnx` files, and through the
live FastAPI service, then asserts:

- `|fused_ondevice − fused_server| ≤ 2` on every fixture,
- identical verdict string on every fixture,
- identical `faceDetected` on every fixture.

Wire it into CI. When it goes red, the extension is lying to users — that is the failure mode
worth the most engineering.

**Ship gate:** parity green on 30/30 fixtures; ms/verdict recorded for wasm and webgpu; the
options page shows the real numbers instead of a promise.

---

### Phase 3 — Explanation on-device (2–3 days)

Grad-CAM needs gradients and ORT Web is forward-only — the review is right. But the server's own
torch-free path already answers this with occlusion saliency, so the honest framing is not
"heatmaps require deep scan", it is **"on-device explains by occlusion, deep scan explains by
Grad-CAM, and the UI says which."**

Port `occlusion_overlay` + `encode_overlay` to TS, faithfully:

- one batched forward of 25 masked copies (grid 5), chunked to fit memory;
- patches filled with **0 after normalisation** ("no information", not "black square");
- `drops = base − occluded`, clipped at 0, divided by peak;
- **`peak ≤ 0` → return null.** No heatmap is a valid answer; a smooth meaningless one is not;
- red/blue blend, `alpha=0.45`, weight = `alpha × cam` so cold regions stay readable.

Verify against Python with a fixture test: same grayscale array in, pixel arrays within ±1.

**Cost:** 26 forwards at 224px. WebGPU → well under a second. Single-thread wasm → seconds. So
explanation is an **explicit "Explain" button**, with a progress indicator and a cancel; never
automatic, and never during a feed scan.

**Anchoring.** The heatmap covers the **face crop**, not the whole photo — same as the server.
The overlay must be positioned over the crop rect mapped back into page coordinates, not
stretched across the image. Stretching it is a silent lie about where the model looked.

**Ship gate:** the on-device overlay for a given image is visually and numerically comparable to
the server's ONNX-path overlay; the UI labels the method and the model source.

---

### Phase 4 — Feeds, Shorts and Reels (4–5 days)

**Discovery is generic. No class names.** `MutationObserver` on `document` plus
`IntersectionObserver` for viewport entry, matching `img`, `video`, and
`[style*="background-image"]`. Filter out chrome by geometry (< 100px on either side), by
`role`/`alt` heuristics for avatars, and by an explicit ignore list of icon/sprite URL patterns.
Per-site adapters exist only to locate the *post container* for badge placement — and the badge
falls back to an absolutely-positioned overlay when no container is recognised.

**Recycled players.** YouTube Shorts and Instagram Reels reuse one `<video>` across clips. State
lives in a `WeakMap<HTMLVideoElement, MediaState>`, and `loadstart` / `emptied` / `currentSrc`
change resets it. Never key state by DOM position or index.

**Frame grab, canvas first.** `drawImage(video)` into an `OffscreenCanvas` works for MSE/`blob:`
sources (same-origin), which covers both target sites — S3 confirms this. On `SecurityError` or
an all-black readback, fall back to `chrome.tabs.captureVisibleTab`, which needs `activeTab`
(user gesture) or a host permission. The review is right that `tabs` alone is not enough.

**Auto mode is on-device only. Full stop.** Auto-uploading whatever a user scrolls past is both a
privacy problem and instantly fatal against a 20/min server budget. This one rule resolves both.
Deep scan stays a deliberate, per-item action.

**Sampling policy:** one frame at `playing`, then every 3s, max 4 per clip. Dedupe by SHA-256 of
a 64×64 grayscale downscale so a paused or static video is not rescanned. Auto mode defaults OFF
and requires the optional host permission, requested with a plain-language explanation.

**Ship gate:** scroll 50 Shorts and 50 Reels with auto mode on — no duplicate scans, no stale
badges on a recycled player, no main-thread frame over 16ms (all inference is in the offscreen
document; the content script only draws).

---

### Phase 5 — Options, packaging, stores (2–3 days)

**Options page:** mode (on-device / deep scan / ask each time), server URL (default prod, editable
so self-hosters can point at `localhost:3000` or a FastAPI at `:8000`), consent state, cache size
and clear, per-site permission toggles via `chrome.permissions.request`, and a **model info
panel** — source string, version, file hashes, fusion weights, thresholds — because "which model
said this" is a question this project answers everywhere else.

**Firefox:** no `chrome.offscreen`, but Firefox MV3 keeps a background *page* with DOM access, so
ORT runs there directly. A build flag `--target=firefox` swaps `offscreen-host.ts` for a
direct-call adapter behind the same interface and emits `manifest.firefox.json` with
`browser_specific_settings.gecko.id`. Edge is Chromium — same package.

**Store submission:** Chrome Web Store $5 one-time developer fee; AMO and Edge are free. Privacy
disclosures: on-device mode collects nothing; deep scan transmits the selected image to the
VerifAI server — declare it, and match the wording to the onboarding screen. No remote code
anywhere (already true by construction).

**Docs:** `extension/README.md` with real measured latency numbers, the parity report, and an
explicit limits section saying what the app's model card says — a "real" verdict means the
detectors found nothing, not that the file is authentic.

---

## 6. Changes needed in this repo (not in the extension)

**All done and verified against a running app.** One deviation, noted below.

- [x] **`app/api/extension/manifest/route.ts`** (new) — `GET` returning
      `{apiVersion, minExtensionVersion, modelVersion, fusionWeights, thresholds, saliencyGrid}`.
      The extension refreshes it daily via `chrome.alarms` and refuses to render an on-device
      verdict whose thresholds have drifted from the server's. Without this, a retuned
      `FUSION_WEIGHTS` on the server silently desyncs every installed extension.
- [x] **`middleware.ts`** — read `X-VerifAI-Client` and give extension traffic its own bucket, so
      a scanning user cannot rate-limit themselves out of the website (and vice versa). Note also
      that the current limiter is an in-memory `Map` in Edge middleware — per-instance, so the
      real limit is fuzzier than 20/min. Worth a comment either way.
- [x] **`app/api/scan/route.ts` + `scripts/inference_server.py`** — accept `?explain=0` and plumb
      it to `analyze(image, explain=False)`. `frame_signals()` already takes `explain`;
      `analyze()` does not. Feed-driven deep scans do not need a heatmap, and skipping it removes
      26 forwards of server CPU per request.
- [x] **`tsconfig.json`** — exclude `extension` (see P3).
- [~] **`.gitignore`** — *deviation.* The plan called for committing the NPR and YuNet ONNX
      files. They are gitignored instead and provisioned by `npm run models`, which pulls them
      from the same URLs `scripts/Dockerfile` already uses. Committing them would add ~28MB to
      every clone and every Vercel deploy for files the deploy never reads — the model service
      builds its own. The build warns when they are absent and the affected detector abstains.

---

## 7. Sequencing and estimate

| | Work | Days | Depends on |
|---|---|---|---|
| P | Prereqs (NPR export, YuNet, fixtures) | 0.5 | — |
| S | Spikes S1–S3 | 1 | — |
| 0 | Scaffold | 1 | — |
| 1 | Server mode + blob fix + cache + consent | 2–3 | 0 |
| 2 | On-device (offscreen, ORT, parity harness) | 4–6 | P, S, 0 |
| 3 | Occlusion explanation | 2–3 | 2 |
| 4 | Feeds / Shorts / Reels | 4–5 | 1, 2 |
| 5 | Options, Firefox, stores | 2–3 | all |
| | **Total** | **17–23 working days** | |

Phase 1 ships standalone — a right-click-to-verify extension is useful with nothing else built.
Phases 2–3 are the differentiator. Phase 4 is the one most likely to overrun, because the two
target sites change without notice; the generic-discovery rule is what keeps that overrun down to
adapter tweaks rather than rewrites.

---

## 8. Standing rules

1. **Never guess.** Model missing, session failed, service down → say so. The app has no
   heuristic fallback anywhere; neither does the extension.
2. **On-device and server must agree.** Parity harness green, or it does not ship.
3. **Auto mode never uploads.**
4. **Every number on screen names its source** — which detectors ran, their weights, and the
   model string, exactly as `/api/scan` already returns them.
5. **Inference never touches the page's main thread** — content scripts draw, the offscreen
   document computes.
