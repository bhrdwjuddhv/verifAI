# Parity fixtures

Drop images here, then:

```
npm run parity:capture     # asks the live server for its verdict on each
npm run parity             # serves the comparison page
```

PLAN.md P4 asks for **30 images**: 10 real, 10 face-swap, 10 fully generated, at least 5 with
no face at all. The mix matters more than the count:

- **no-face images** are the only ones that exercise NPR voting alone, and the abstain path;
- **face images** are the only ones that exercise detect → crop at 0.35 margin → classify;
- **fully generated faces** are the case the face classifier was never trained to catch and
  NPR exists to cover, so a set without them cannot show the fusion doing its job.

Use full-resolution originals, not thumbnails or screenshots. NPR reads the resampling
fingerprint a generator's decoder leaves behind, and re-encoding overwrites it — a downscaled
Google Images thumbnail of an AI picture will often score as real for that reason alone.

The images themselves are gitignored: they are large, and most are not ours to redistribute.
