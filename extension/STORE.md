# Store submission

Everything a reviewer will ask for, written once. Chrome Web Store charges a **$5 one-time**
developer fee; AMO and the Edge Add-ons store are free. The Edge package is the Chrome package.

```bash
npm run pack            # verifai-<version>-chrome.zip   (Chrome + Edge)
npm run pack:firefox    # verifai-<version>-firefox.zip
```

`pack` refuses to produce an archive containing preview pages, captured fixtures or source
maps, so a local debugging session cannot end up in a store upload by accident.

---

## Listing copy

**Name** — VerifAI — AI media verification

**Short description** (132 char limit)
> Right-click any image or video to check whether it was AI-generated or manipulated, and see
> which detector said so.

**Full description**

> VerifAI checks whether an image was AI-generated or manipulated, and shows you the working.
>
> Right-click any image and choose "Verify with VerifAI". You get a verdict, the score each
> detector produced, the weights that combined them, and the name of the model that ran — so
> you can judge the number instead of trusting it.
>
> **Two ways to scan.** On-device mode runs the bundled models locally and uploads nothing.
> Deep scan sends the image to a VerifAI server you configure — your own, if you self-host.
> You choose, and the extension asks before it uploads anything the first time.
>
> **It tells you when it does not know.** If no face is found, the face classifier abstains
> rather than guessing. If a detector is missing, it says so instead of being replaced by a
> number. A "real" verdict means the detectors found nothing — not that the file is authentic,
> and the extension says that too.
>
> **Explain a score.** On-device results can show where the model looked, measured by hiding
> each region in turn and watching the score move.
>
> Open source, no analytics, no accounts, no remote code.

---

## Permission justifications

Reviewers ask for these one by one. Each is the narrowest thing that makes a feature work.

| Permission | Why |
|---|---|
| `contextMenus` | Adds the single "Verify with VerifAI" item to the right-click menu on images and video. It is the entire entry point. |
| `activeTab` | When you invoke the menu item, the image may be a `blob:` URL that exists only inside that page — WhatsApp Web and Instagram both do this — and cannot be fetched from outside it. This grants read access to that one tab, for that one action. |
| `scripting` | Two uses: reading that `blob:` image from the page, and drawing the result badge over the image you scanned. Nothing is injected until you ask for a scan. |
| `storage` | Your settings, and a cache of verdicts keyed by the hash of the file so re-checking the same image sends nothing. |
| `offscreen` | Chrome only. ONNX Runtime needs a document with WebGPU and a lifetime longer than a service worker's; this is the API for exactly that. Firefox's build does not request it. |

### Content Security Policy — `'wasm-unsafe-eval'`

The manifest relaxes the default extension CSP by exactly one token:

```
script-src 'self' 'wasm-unsafe-eval'; object-src 'self'
```

Reviewers ask about this, and the answer is narrow: MV3's default `script-src 'self'` forbids
`WebAssembly.instantiate` outright, and the detectors run on ONNX Runtime's WebAssembly build.
`'wasm-unsafe-eval'` permits compiling WebAssembly and **nothing else** — unlike
`'unsafe-eval'`, it does not enable `eval`, `new Function`, or inline script.

Every `.wasm` and `.onnx` file is inside the package. Nothing is fetched, and the ONNX Runtime
build was chosen specifically because the default distribution contains an `eval` this policy
would — correctly — refuse to run.

### Optional host permissions — `http://*/*`, `https://*/*`

**Not requested at install.** Nothing is granted until you act, and each grant is a separate
prompt naming a specific site.

Three things need host access, and none of them can be predicted at packaging time:

1. **The VerifAI server.** It is user-configurable — a self-hoster points it at their own
   deployment or `localhost`. Requested once, for that origin, on the consent screen.
2. **The image you asked about.** Media lives on arbitrary CDNs. When a scan needs a download
   the extension has no access to, it asks for that origin and nothing else.
3. **Auto-scan sites.** Requested per site from the options page, only when you switch
   auto-scan on, and revoked the moment you switch it off.

If a narrower declaration is required, the two auto-scan sites can be pinned to
`https://*.youtube.com/*` and `https://*.instagram.com/*`; the other two uses genuinely cannot
be enumerated in advance.

---

## Privacy disclosures

**Data collected: none.** No analytics, no telemetry, no accounts, no identifiers, no
third-party services. The extension talks to exactly one server: the one the user configured.

| Question | Answer |
|---|---|
| Personally identifiable information | No |
| Health, financial, authentication information | No |
| Personal communications, location, web history | No |
| User activity (clicks, mouse position, keystrokes) | No |
| **Website content** | **Yes — in deep scan mode only** |

**Website content, stated plainly.** Scans run on the user's machine by default and upload
nothing. A "deep scan" — offered for video, where the server samples every frame — uploads the
file the user selected to the VerifAI server they configured, after a confirmation naming that
server. It is never automatic. Nothing else is sent: no page URL, no cookies, no browsing history, no identifiers.
The user accepts this on a first-run screen that says so before any upload happens, and
on-device mode does not upload at all.

**Auto-scan** never uploads under any setting. It runs only when the scan mode is on-device,
and the service worker re-checks that at the moment of every scan.

**Remote code: none.** Every model, the ONNX Runtime WebAssembly and all JavaScript are in the
package. There is no CDN load, no `eval`, and no hosted-code execution path — the non-bundled
WebGPU build of ONNX Runtime was chosen specifically because the default entry point contains
an `eval` that MV3 forbids.

Required single purpose statement:

> Verifying whether an image or video the user selects was AI-generated or manipulated.

---

## Publishing, step by step

### 1. The default is already on-device — know why it matters

Nothing is uploaded unless a user explicitly takes up an offer of a deep scan, which names the
destination and asks first. That is what lets this listing say it collects nothing by default,
and it means a public install costs you no compute and hands you no strangers' images.

If you change `DEFAULTS.mode` to `'server'` before publishing, revisit the privacy section
below: every install would then upload to your deployment, and the disclosure has to lead with
that rather than qualify it.

### 2. Register as a developer — $5, once

<https://chrome.google.com/webstore/devconsole> — sign in with the Google account that should
own the extension, pay the one-time $5 registration fee, and verify your email. Google may
also ask for identity verification before your first public listing; do it early, it can take
a day or two on its own.

Use an account you are willing to keep. Transferring an extension later is possible but
awkward, and the account name appears on the listing.

### 3. Publish a privacy policy

Required for anything that handles user data, and deep scan does. You already run a site — add
a page at `https://<your-host>/privacy` covering: what is collected (nothing, unless deep scan
is used), what deep scan transmits (the selected image, to the configured server), retention,
and a contact address. The URL goes in the listing's Privacy tab.

### 4. Take screenshots

1280×800 or 640×400, PNG or JPEG, between one and five. They are the listing, so make them
show the thing that is actually distinctive:

- a verdict with the per-detector breakdown and the fusion weights visible
- an occlusion heatmap over a face
- the options page with on-device selected

### 5. Build and upload

```bash
cd extension && npm run pack
```

Dashboard → **Items** → **Add new item** → upload `verifai-<version>-chrome.zip`. The zip must
have `manifest.json` at its root, which `pack` guarantees.

### 6. Fill the listing

Copy from the sections above. Category: **Productivity**. Fill every permission justification
— reviewers reject submissions that leave them blank far more often than they reject the
permissions themselves.

### 7. Distribution

Set visibility to **Public** and pick your regions. Public is what makes it searchable;
Unlisted works by link only, which is a good way to test the whole flow with real users first.

### 8. Submit, and expect a wait

Simple extensions clear review in a few days. **Expect longer here**: broad optional host
permissions and a CSP relaxation both add scrutiny, and each round trip costs days. Answer
their questions with the justifications above rather than rewriting them.

Once approved it is live immediately, though store *search* takes a while to index and rank —
share the direct link in the meantime.

### 9. Updating

Bump `version` in `extension/package.json` (the manifest takes it from there), `npm run pack`,
upload as a new package. Updates go through review too, usually faster. Users get it
automatically within a few hours.

### The other two stores

- **Edge**: <https://partner.microsoft.com/dashboard/microsoftedge> — free, takes the same
  Chrome package unchanged.
- **Firefox**: <https://addons.mozilla.org/developers/> — free, upload
  `verifai-<version>-firefox.zip`. AMO requires reviewable source for anything minified; this
  build is not minified, and the repo is the source.

---

## Pre-submission checklist

- [ ] `npm run typecheck && npm run selfcheck && npm run build` all clean
- [ ] `npm run parity` is green, or the gap is documented in the listing's limitations
- [ ] Version bumped in `package.json` (the manifest takes it from there)
- [ ] Screenshots: 1280×800 — a verdict with the detector breakdown, an explanation heatmap,
      the options page showing on-device mode
- [ ] Privacy policy URL published and reachable
- [ ] Firefox package tested via `about:debugging`
- [ ] Listing states the limits: accuracy against unseen generators is not yet measured, and
      confidence is uncalibrated
- [ ] Default scan mode decided deliberately (see step 1) — it decides whether the listing
      declares any data collection at all
- [ ] Loaded the packed zip unpacked once and confirmed Options -> Probe reports
      `WebAssembly: allowed`; a manifest that never got reloaded fails every on-device scan

## Known limitations to declare

Both are already in the product's own copy; the listing should not be more confident than the
extension is.

- A "real" verdict means the detectors found nothing, not that the file is authentic.
- Confidence is uncalibrated — a ranking, not a probability.
- Re-encoded images (a search-result thumbnail, a re-compressed screenshot) weaken the
  whole-image detector badly, because it reads a resampling fingerprint that re-encoding
  overwrites.
