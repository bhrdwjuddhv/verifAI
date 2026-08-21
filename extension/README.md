# VerifAI browser extension

Right-click any image or video → **Verify with VerifAI** → the same verdict the website gives,
with the same per-detector breakdown, in place.

All five build phases are implemented. The status table below is the current state.

---

## Live Guard (live calls)

Monitors a call **playing in a browser tab** — Google Meet, Discord web, Teams web, Zoom web —
and shows a live trust score, a per-window timeline, and a warning when synthetic voice signal
persists. It never ends a call.

**What it cannot do, and will not pretend to:** a native desktop app (Zoom, Teams, WhatsApp,
Discord's own client) and a cellular phone call are both invisible to an extension. No browser
API exposes their audio. If your call is not in a tab, Live Guard is not running, and the
overlay will not appear.

Other limits, stated rather than buried:

- Audio windows go to **your** VerifAI backend (`/api/live/audio-window`), which runs this
  project's trained voice model. Nothing is sent anywhere else. On-device audio is not
  implemented — see below.
- Verdicts are refused unless `modelSource` names one of this project's trained models. A
  score from the Hugging Face fallback is shown as a refusal, not as a number.
- `tabCapture` is requested in the manifest; the call platforms are **optional** host
  permissions, granted per platform when you start monitoring.
- Capturing tab audio mutes the tab unless the stream is played back — Live Guard routes it
  through at unity gain, so you still hear your call.

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

**Scans run on your machine.** The three bundled detectors are the same ones the server runs,
and parity against it is exact on still images — so there is no trade to weigh and nothing to
choose. Deep scan is not a second mode you live in; it is offered per scan where the server
genuinely does more (sampling every frame of a video), and it asks first, naming the
destination. A detector that cannot apply **abstains and says so** — it is never replaced by
a guess.

## Running it on your machine

**You need:** Node 20+, and Python 3.10+ with two packages. Python is used once, to convert the
NPR detector — the model is published only as a PyTorch checkpoint, but the converter here
needs neither PyTorch nor Docker.

```bash
pip install numpy onnx
```

Then, from the repo root:

```bash
cd extension && npm install && npm run models && npm run build
```

`npm run models` fetches the two detectors that are not in git — YuNet and NPR — from the same
URLs `scripts/Dockerfile` already uses to provision the server, and converts NPR to ONNX. The
face classifier is committed, so it needs nothing. Expect about 6MB of downloads.

Load it in Chrome:

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. **Load unpacked** → select `extension/dist` — the folder itself, not a file inside it

Confirm it worked: open the extension's **Options** and press **Probe this machine**. It should
report `WebAssembly: allowed` and a backend of `webgpu` or `wasm`. If WebAssembly says blocked,
the manifest has not been re-read — press reload on the extension card.

Then right-click any image and choose **Verify with VerifAI**.

### After you change anything

```bash
npm run build
```

then press **reload** on the extension card. Chrome picks up rebuilt code on its own for some
surfaces but **never** re-reads `manifest.json` without that reload — which is how a
permission or CSP change appears to do nothing. If the popup and the background worker ever
fall out of step, the popup says so in an amber banner rather than failing mysteriously.

### If something looks wrong

| Symptom | Cause |
|---|---|
| `WebAssembly: blocked` in the probe | The manifest was not re-read. Reload the extension. |
| "does not handle …" on a button | Same thing: the worker is older than the popup. |
| `Extension context invalidated` in the error log | Scripts orphaned by a *previous* reload. Refresh the affected tabs; they stop on their own. |
| On-device verdict says a detector is missing | `npm run models` did not finish — re-run it and check the Python step. |

Firefox: `npm run build:firefox`, then load `dist/manifest.json` via `about:debugging`.

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
