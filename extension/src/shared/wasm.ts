/** Runtime check for the one manifest setting that silently disables every detector. */
/**
 * Can this context compile WebAssembly at all?
 *
 * MV3's default `script-src 'self'` forbids it outright, and the manifest has to declare
 * `'wasm-unsafe-eval'` to allow it. Chrome only re-reads the manifest when the extension is
 * reloaded, so an install that predates that change fails every on-device scan with a
 * CompileError that names ONNX Runtime rather than the real cause. Eight bytes — the shortest
 * valid module there is — answer it definitively.
 */
export function wasmAllowed(): { ok: boolean; reason?: string } {
  try {
    new WebAssembly.Module(new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]));
    return { ok: true };
  } catch (err) {
    const message = String((err as Error)?.message ?? err);
    return {
      ok: false,
      reason: /content security policy/i.test(message)
        ? "blocked by the extension's Content Security Policy — reload the extension at chrome://extensions to pick up the current manifest"
        : message,
    };
  }
}
