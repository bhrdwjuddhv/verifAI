/**
 * Magic-byte file content validation.
 * Verifies actual file content matches declared type by checking file signatures (magic bytes).
 * Prevents disguised files (e.g., a .exe renamed to .jpg) from being processed.
 */

interface FileSignature {
  mimeType: string;
  category: 'image' | 'video' | 'audio';
  // Each signature is an array of { offset, bytes } checks
  signatures: Array<{ offset: number; bytes: number[] }>;
}

const FILE_SIGNATURES: FileSignature[] = [
  // JPEG: FF D8 FF
  {
    mimeType: 'image/jpeg',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0xff, 0xd8, 0xff] }],
  },
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  {
    mimeType: 'image/png',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] }],
  },
  // GIF87a / GIF89a: 47 49 46 38
  {
    mimeType: 'image/gif',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0x47, 0x49, 0x46, 0x38] }],
  },
  // WebP: RIFF....WEBP
  {
    mimeType: 'image/webp',
    category: 'image',
    signatures: [
      { offset: 0, bytes: [0x52, 0x49, 0x46, 0x46] }, // RIFF
      { offset: 8, bytes: [0x57, 0x45, 0x42, 0x50] }, // WEBP
    ],
  },
  // BMP: 42 4D
  {
    mimeType: 'image/bmp',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0x42, 0x4d] }],
  },
  // TIFF (little-endian): 49 49 2A 00
  {
    mimeType: 'image/tiff',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0x49, 0x49, 0x2a, 0x00] }],
  },
  // TIFF (big-endian): 4D 4D 00 2A
  {
    mimeType: 'image/tiff',
    category: 'image',
    signatures: [{ offset: 0, bytes: [0x4d, 0x4d, 0x00, 0x2a] }],
  },
  // MP4/M4A (ftyp box): 00 00 00 xx 66 74 79 70  (ftyp at offset 4)
  {
    mimeType: 'video/mp4',
    category: 'video',
    signatures: [{ offset: 4, bytes: [0x66, 0x74, 0x79, 0x70] }], // "ftyp"
  },
  // AVI: RIFF....AVI
  {
    mimeType: 'video/avi',
    category: 'video',
    signatures: [
      { offset: 0, bytes: [0x52, 0x49, 0x46, 0x46] }, // RIFF
      { offset: 8, bytes: [0x41, 0x56, 0x49, 0x20] }, // AVI
    ],
  },
  // WebM/MKV: 1A 45 DF A3
  {
    mimeType: 'video/webm',
    category: 'video',
    signatures: [{ offset: 0, bytes: [0x1a, 0x45, 0xdf, 0xa3] }],
  },
  // MOV (same ftyp as MP4 but with 'qt' brand - detected via ftyp)
  // Already covered by MP4 ftyp detection

  // WAV: RIFF....WAVE
  {
    mimeType: 'audio/wav',
    category: 'audio',
    signatures: [
      { offset: 0, bytes: [0x52, 0x49, 0x46, 0x46] }, // RIFF
      { offset: 8, bytes: [0x57, 0x41, 0x56, 0x45] }, // WAVE
    ],
  },
  // MP3 (ID3v2 tag): 49 44 33
  {
    mimeType: 'audio/mpeg',
    category: 'audio',
    signatures: [{ offset: 0, bytes: [0x49, 0x44, 0x33] }], // "ID3"
  },
  // MP3 (sync word): FF FB or FF FA or FF F3 or FF F2
  {
    mimeType: 'audio/mpeg',
    category: 'audio',
    signatures: [{ offset: 0, bytes: [0xff, 0xfb] }],
  },
  {
    mimeType: 'audio/mpeg',
    category: 'audio',
    signatures: [{ offset: 0, bytes: [0xff, 0xf3] }],
  },
  // FLAC: 66 4C 61 43
  {
    mimeType: 'audio/flac',
    category: 'audio',
    signatures: [{ offset: 0, bytes: [0x66, 0x4c, 0x61, 0x43] }],
  },
  // OGG: 4F 67 67 53
  {
    mimeType: 'audio/ogg',
    category: 'audio',
    signatures: [{ offset: 0, bytes: [0x4f, 0x67, 0x67, 0x53] }],
  },
];

function matchesSignature(bytes: Uint8Array, sig: FileSignature): boolean {
  return sig.signatures.every((check) => {
    if (check.offset + check.bytes.length > bytes.length) return false;
    return check.bytes.every((b, i) => bytes[check.offset + i] === b);
  });
}

export interface FileValidationResult {
  valid: boolean;
  detectedMimeType: string | null;
  detectedCategory: 'image' | 'video' | 'audio' | 'unknown';
  reason?: string;
}

/**
 * Validate that a file's actual content (magic bytes) matches what the declared type claims.
 *
 * @param buffer - Raw file buffer (at least first 16 bytes needed)
 * @param declaredCategory - 'image', 'video', or 'audio' as declared by the client
 */
export function validateFileContent(
  buffer: ArrayBuffer,
  declaredCategory: 'image' | 'video' | 'audio'
): FileValidationResult {
  const bytes = new Uint8Array(buffer.slice(0, 64)); // Only need first 64 bytes for magic checks

  if (bytes.length < 4) {
    return {
      valid: false,
      detectedMimeType: null,
      detectedCategory: 'unknown',
      reason: 'File is too small to validate (less than 4 bytes).',
    };
  }

  // Find a matching signature
  for (const sig of FILE_SIGNATURES) {
    if (matchesSignature(bytes, sig)) {
      // Found a match — check if it matches the declared category
      if (sig.category === declaredCategory) {
        return {
          valid: true,
          detectedMimeType: sig.mimeType,
          detectedCategory: sig.category,
        };
      }

      // Mismatch — file content doesn't match what was declared
      return {
        valid: false,
        detectedMimeType: sig.mimeType,
        detectedCategory: sig.category,
        reason: `File content is actually ${sig.mimeType} (${sig.category}), but was declared as ${declaredCategory}. Upload rejected for security.`,
      };
    }
  }

  // No signature matched at all — unknown/suspicious file type
  return {
    valid: false,
    detectedMimeType: null,
    detectedCategory: 'unknown',
    reason: `Unable to verify file content type. The file does not match any known ${declaredCategory} format signature. Only genuine image/video/audio files are accepted.`,
  };
}

/** Max upload size: 50 MB */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/**
 * Check if a file exceeds the maximum upload size.
 */
export function isFileTooLarge(sizeBytes: number): boolean {
  return sizeBytes > MAX_UPLOAD_BYTES;
}
