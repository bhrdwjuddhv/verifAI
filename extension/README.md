# VerifAI browser extension

Right-click any image or video → **Verify with VerifAI** → the same verdict the website gives,
with the same per-detector breakdown, in place.

All five build phases are implemented. The status table below is the current state.

---

## Status

| Phase | State |
|---|---|
| 0 — Scaffold, MV3 manifest, two-format Vite build | **done** |
| 1 — Right-click verify, `blob:` reads, SHA-256 cache, queue, consent | **done** |
| 2 — On-device inference in an offscreen document | **done, face-only** — NPR needs an export |
| 3 — Occlusion saliency on-device | **done** — explicit action, never automatic |
| 4 — Feeds, Shorts, Reels | **built** — off by default, on-device only |
| 5 — Firefox target, store packaging | **done** — see [STORE.md](STORE.md) |

Both scan modes work. Deep scan uploads to your VerifAI server; on-device decodes and scores
locally and uploads nothing. A detector that is not bundled **abstains and says so** in the
verdict's notes — it is never replaced by a guess.

## Build and load

```bash
cd extension && npm install && npm run models && npm run build
```

`npm run models` fetches the detectors the deployed service uses (see
[fetch-models.mjs](scripts/fetch-models.mjs)); weights are copied into `dist/` at build time
rather than committed here.

Then in Chrome: `chrome://extensions` → enable **Developer mode** → **Load unpacked** →
select `extension/dist`.

First run opens a setup page that asks one thing: **where scans run**. On-device is the
default and needs no permission at all — nothing leaves the machine, so there is nothing to
grant. Choosing deep scan reveals the server field (default `https://verif-ai-blue.vercel.app`;
point it at `http://localhost:3000` for development) and asks Chrome for access to that one
address. The page describes what is actually bundled by reading the build's own model
manifest, so it cannot claim a detector that is not in the package.

Firefox build: `npm run build:firefox`, then load `dist/manifest.json` via `about:debugging`.

## On-device mode

Everything runs in an offscreen document — the only extension context with `navigator.gpu`
and a lifetime longer than the service worker's ~30s idle eviction.

| | |
|---|---|
| Backend | WebGPU when an adapter exists, wasm otherwise |
| Threads | **none** — extension pages get no cross-origin isolation, so no `SharedArrayBuffer` |
| Face detector | YuNet ONNX, letterboxed into its fixed 640×640 input |
| Face classifier | the repo's `detector.onnx`, fp32, flip-TTA, T=0.7193 |
| NPR | not bundled yet — needs a torch export, see `npm run models` |
| Measured | 3.7s cold (21MB wasm + 16MB session), 1.0–1.4s warm, on a 512×384 PNG |

**The manifest must allow WebAssembly.** MV3's default `script-src 'self'` blocks
`WebAssembly.instantiate` itself, so the manifest declares
`script-src 'self' 'wasm-unsafe-eval'`. Avoiding `eval` in the JavaScript is necessary but not
sufficient — without that token every on-device scan fails with a `CompileError` and no model
loads at all. `scripts/preview.mjs` serves its harness pages under the manifest's own CSP for
exactly this reason: a harness on a permissive localhost policy will happily run WebAssembly
the real extension is forbidden to compile, and that gap let this reach a user once already.

Parity with the server is the point, and three pieces exist only to protect it:

- **[resample.ts](src/shared/resample.ts)** is a port of Pillow's `Resample.c` — the same
  triangle filter with support scaled by the reduction factor, the same 22-bit fixed-point
  coefficients, the same clip to uint8 between passes. A canvas `drawImage` downscale is a
  different filter, and the difference is several points of fused score.
- **[detect.ts](src/shared/detect.ts)** ports the fusion, verdict bands, label mapping and
  frequency score literally, including the renormalisation that stops a missing detector
  voting "50".
- **`createImageBitmap(..., { imageOrientation: 'none' })`**, because the browser applies EXIF
  rotation by default and `PIL.Image.open` does not.

Two known divergences, both documented in the code: YuNet runs at a fixed 640×640 here while
the service detects at native resolution, and NPR does not vote until it is exported.

## Explaining a score

Grad-CAM needs gradients and ONNX Runtime is forward-only, so on-device explanation is
**occlusion saliency** — the same thing the service's own torch-free path does. Each of 25
patches is hidden in turn and the drop in P(fake) measured; the map is what the model
*depended on*, not what it activated on. The popup labels which method produced a heatmap, so
an on-device map is never passed off as Grad-CAM.

It is an explicit **"Explain this score"** button, not something every scan pays for: 26
forward passes, measured at **897ms on CPU** via onnxruntime-node. WebGPU is faster; the popup
prints the real figure next to every on-device result.

Three details are ported literally from `occlusion_overlay` and `encode_overlay`, and all
three are the difference between an honest picture and a plausible one:

- patches are filled with **0 after normalisation** — the dataset mean, i.e. "no information",
  not a black square, which would itself be a strong feature;
- a peak of zero returns **null**. If hiding any region leaves the score unchanged there is
  nothing honest to draw, and the note says exactly that;
- the blend is weighted by the map itself (`weight = alpha * cam`), so cold regions are left
  untouched rather than washed flat.

The overlay covers the **face crop**, not the whole photo — the same as the server. It is
never stretched across the original image, because that would be a silent lie about where the
model looked.

## Video, Shorts and Reels

**A right-click never reaches a YouTube player.** The page replaces Chrome's context menu with
its own, so the "Verify with VerifAI" item is not there to click — on Shorts and Reels the
manual path simply does not exist. The popup's **"Scan the video playing here"** is the way in:
clicking the toolbar icon grants `activeTab`, which is enough to ask the page for the current
frame. It works on any player that swallows the context menu, and it respects the current scan
mode like anything else.

The popup also says, in one line, whether auto-scan is watching the current tab and why not if
it isn't. Silence was the worst possible answer to "why is nothing happening".

## Auto-scan feeds

Off by default. Enable it in options and grant a site, and the extension watches that feed and
scores what scrolls past. **It only ever runs on-device** — auto-uploading whatever a user
scrolls past is both a privacy problem and instantly fatal against a 20/min server budget, so
switching to deep scan makes it inert. The worker re-checks that at the moment of every scan,
not just at registration.

No content script is registered until a site is granted, and it is unregistered the moment the
grant is revoked or auto-scan is switched off — so what the options page says the extension can
reach is always what it can reach.

**Discovery has no class-name selectors.** Instagram's and YouTube's class names are obfuscated
and change without notice. What does not change: a post contains an `img` or a `video`, it is
large, and it enters the viewport. Rejection is by *role* — an asset under 100px either
rendered or natural is an avatar or an icon, and `/icons/`, `/sprite`, `/emoji/` are never post
content.

Badges are **one per media element**, not one per scan. Keying them by scan id was wrong in a
way that only appears on a feed: auto-scan samples four frames per clip, so four badges
appeared per Short, and once the recycled player swapped source they stopped matching their
anchor and every one parked in the same corner. Scrolling built a pile. Now a scan updates the
badge belonging to its element, a clip change takes the previous verdict down, an auto badge
with nothing left to point at is dropped rather than parked, and the total is capped.

**A reloaded extension orphans its content scripts.** They keep running with no way home:
`chrome.runtime.sendMessage` throws `Extension context invalidated` *synchronously* — so a
`.catch()` on the promise never sees it — and `chrome.runtime` itself becomes undefined. The
scripts must stop, not just swallow the error, or every open feed tab keeps sampling video on
a 3-second timer for the life of the tab. `src/content/runtime.ts` guards every send, and
polls `chrome.runtime.id` every 5s so a script whose frames all deduplicate still notices;
teardown cancels the observers, the timers and the badges.

Four bugs the harness caught, all of which would have looked like "it works on my test page":

- **Lazy-loaded images were never found.** Feeds put the `<img>` in the DOM and set `src`
  later, so the intersection callback arrives while the element is empty. Dropping it there is
  terminal: the element is already intersecting, so the observer never fires for it again.
- **A recycled player stopped after one clip.** Shorts and Reels swap the source on one
  `<video>` through MSE without necessarily firing `playing` again, so resetting state on
  `loadstart` without re-arming the timer meant every clip after the first was skipped.
- **Badges piled up while scrolling** — the one above, now covered by four assertions that
  scan the same element repeatedly and check exactly one badge exists.
- **An orphaned script never stopped.** The harness deletes `chrome.runtime.id` — exactly what
  Chrome does on reload — and asserts that sampling ceases and the badges come down. The first
  version of the fix passed the sampling half and failed the badge half, because teardown only
  ran when something tried to send.

Frames are captured in the page — canvas-first, which needs no CDN permission at all — one at
playback start then every 3s, four per clip, deduped by the SHA-256 of a 64×64 grayscale
reduction so a still shot is scanned once. They are encoded as **PNG, never JPEG**: the frame
is already lossy from the codec, and a second lossy pass would overwrite the resampling
fingerprint NPR reads. DRM-protected media taints the canvas, which is detected and reported
rather than scored as a black rectangle.

Feed *images* are a different story: they live on a CDN the extension has no access to. Those
are attempted through the worker and skipped if the CDN refuses, rather than interrupting a
scroll with a permission prompt.

Auto-scan requires **three** things at once, and it is off by default: the setting enabled,
the scan mode set to on-device, and the site granted. Any one missing and nothing happens —
which is why the popup states which one it is.

```bash
node scripts/preview.mjs   # then open /preview-feed.html
```

That harness builds a mixed feed — content image, upscaled avatar, icon URL, thumbnail, and a
live `<video>` fed by `canvas.captureStream()` — and asserts all of the above, including a
simulated clip swap on the same element. 11 assertions, no network, no YouTube.

## Parity harness

On-device and server must agree — that is the rule this whole port exists to keep. This measures it.

```bash
npm run parity:capture && npm run parity
```

`parity:capture` POSTs every image in `fixtures/` to the live server and records its verdict,
paced at 12/min so it stays inside the 20/min budget `middleware.ts` enforces. `parity` then
serves a page that runs the same images through the real on-device pipeline — in a browser, so
the canvas decode being tested is the one that ships — and compares them signal by signal.

It compares **per detector**, not just the fused score, and that is the important design
choice. A detector present on one side only makes fused scores incomparable by construction:

| Fixture | Face | NPR | Freq | Fused | Verdict | Face? | Status |
|---|---|---|---|---|---|---|---|
| smoke-icon.png | — / absent | 0 / **absent** | 0 / 0 | 0 / absent | Real / Uncertain | false / false | npr server-only |

Read naively that is a failure — different verdict, no comparable score. It is not. The server
has NPR and this build does not, so the numbers measure the gap rather than the port. The
harness says so instead of printing a red Δ and sending you hunting.

What the same row *does* prove: `frequencyScore` matched **exactly** (0/0) on a real image,
which is the FFT, the Pillow fixed-point luma and the resampler agreeing with numpy and PIL;
and `faceDetected` matched, which is the YuNet port agreeing with OpenCV's.

Green parity needs NPR exported — see `npm run models`.

## How a scan runs

```
right-click ──► service worker ──► read bytes ──► SHA-256 ──► cache hit? ──► done
                     │                  │                          │
                     │                  │ blob: / data:            │ miss
                     │                  └─► page-side read         ▼
                     │                      (base64 back)     token bucket 12/min
                     │                                             │
                     └────────── badge in page ◄── verdict ◄── POST /api/scan
```

Four decisions worth knowing, because they are not obvious and they are load-bearing:

**No declared content script, no install-time host permissions.** The badge script is injected
with `chrome.scripting` only when a scan is asked for, using the `activeTab` grant the
context-menu click provides. The extension holds access to no site until you use it.

**`blob:` media is read inside the page.** WhatsApp Web and Instagram serve images as
`blob:` URLs, which the service worker cannot fetch. The bytes come back as base64, because
Chrome's `runtime.sendMessage` JSON-serialises and an `ArrayBuffer` would arrive as `{}`.

**The POST lives in the service worker.** Fetches from there are exempt from CORS for granted
origins, custom headers and all. The same fetch from a content script runs under the page's
CORS and fails on most CDNs.

**Scans queue at 12/min.** `middleware.ts` caps `/api/*` at 20 requests per minute per IP —
a budget shared with whoever is browsing the site from the same address. Running at the
ceiling would rate-limit you out of your own website.

## Layout

```
manifest.chrome.json   Chrome's manifest; the Firefox one is derived at build time
scripts/build.mjs      two Vite passes — IIFE for the injected script, ESM for everything else
scripts/gen-icons.mjs  the toolbar icons, drawn with node:zlib and no image library
scripts/selfcheck.mjs  47 assertions over the real ONNX, running the extension's own source
scripts/fetch-models.mjs  provisions the detectors from the Dockerfile's own sources
src/background/        service worker: menus, media, api, cache, queue, scan state, offscreen host
src/content/           the in-page badge (shadow DOM, adopted stylesheet)
src/offscreen/         the only place inference happens: ORT sessions, scoring, capability probe
src/shared/            scoring maths, Pillow-compatible resampling, YuNet decode, protocol,
                       settings, and the scan types imported from the app
src/popup|options|onboarding/
```

## Packaging

```bash
npm run pack            # verifai-<version>-chrome.zip   (Chrome + Edge)
npm run pack:firefox    # verifai-<version>-firefox.zip
```

19.5MB compressed, most of it the 21MB ONNX Runtime wasm and the 16MB face model. The ZIP
writer is `node:zlib` and about 100 lines, for the same reason the icons are: a release step
that depends on `archiver` being installed correctly is a release step that breaks on someone
else's machine.

`pack` **refuses to produce an archive** containing preview pages, captured fixtures or source
maps, so a debugging session cannot end up in a store upload.

Firefox is a real second target, not a manifest rewrite. It has no `chrome.offscreen` — but it
does not need one: its background is an event *page*, so it already has a DOM, `OffscreenCanvas`
and `navigator.gpu`, and the scorer runs there directly. `src/background/host.chrome.ts` and
`host.firefox.ts` implement one interface, the build aliases `#host` to the right one, and
nothing above them knows which browser it is on. The Firefox package also drops the offscreen
document rather than shipping a page nothing can open.

[STORE.md](STORE.md) carries the listing copy, the per-permission justifications a reviewer
asks for one by one, and the privacy disclosures — including the one that matters: deep scan
uploads the selected image, on-device mode uploads nothing, and auto-scan never uploads under
any setting.

## Relationship to the website

The extension is a separate toolchain in the same repo and **cannot affect the deployed site**:

- its own `package.json` and `node_modules`; nothing is added to the app's dependencies;
- `tsconfig.json` at the repo root excludes `extension/`, so `next build` never typechecks it;
- Tailwind's `content` globs only cover `app/` and `components/`;
- the Vite build pins an empty PostCSS config, so it does not pick up the app's Tailwind;
- model weights and ORT wasm are **copied in at build time**, never committed here, so the
  Vercel deploy does not carry a second copy of a 16MB ONNX.

It does share one thing on purpose: `src/shared/scan-types.ts` imports `ScanResult` from
`lib/store.ts` as a type. Change the `/api/scan` response shape and `npm run typecheck` fails
here instead of the popup silently rendering a blank verdict.

## What it deliberately does not do

- Guess. Model missing, service down, file unreadable — each says so, distinctly. A 503 from
  the model service and a 415 rejected file are different messages, never "scan failed".
- Upload anything in on-device mode.
- Claim a "real" verdict means authentic. It means the detectors found nothing.

## Checks

```bash
npm run typecheck && npm run selfcheck && npm run build
```

`selfcheck` is the counterpart to `python scripts/inference_server.py --selfcheck`: 47
assertions over the real ONNX files, covering the things that fail *silently* — an
unnormalised resample, a YuNet anchor grid indexed the wrong way, a missing detector dragging
the fused score toward 50. It runs the extension's own TypeScript under Node, so there is no
second implementation to keep in sync.

`node scripts/preview.mjs` serves the popup, options and onboarding pages with a stubbed
`chrome.*`, plus an offscreen harness that runs the real ONNX pipeline in an ordinary tab.
