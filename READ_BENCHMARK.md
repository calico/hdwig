# Does a float16 lookup speed up reads?

Question: should `Track.bins` unscale float16 by looking each bit pattern up in a table, rather than dividing?

Answer: yes. After the first read in a process, the lookup is as fast or faster on every read measured, NaN-bearing or not. The cost is a one-time ~1.7 ms table build per (scale, output dtype).

## Results (ms, median)

| Read | Data | Division | Guarded lookup | Cached lookup |
|---|---|---:|---:|---:|
| First open + 128 bp in process | finite | 3.10 | 3.20 | 4.93 |
| Open + 128 bp | finite | 1.47 | 1.48 | 1.47 |
| Open + whole contig, float16 | finite | 48.6 | 33.6 | 29.5 |
| Open + whole contig, float16 | 1% NaN | 49.5 | 50.7 | 27.4 |
| Open + whole contig, float32 | 1% NaN | 29.8 | 29.8 | 27.4 |
| In RAM, whole contig, float16 | finite | 27.2 | 9.29 | 7.87 |
| In RAM, whole contig, float16 | 1% NaN | 27.5 | 28.9 | 7.90 |
| In RAM, whole contig, float32 | 1% NaN | 10.4 | 10.2 | 8.10 |

Float16 output gains most: float16 division runs in software. Float32 output gains ~20%. The guarded lookup has no speedup on NaN-bearing data, where NaN marks "no data" and is common.

## Method

One contig of 4,194,304 exponential float32 values (mean 3, NumPy seed 441), written with default settings (float16, zstd); the NaN track sets every 100th value to NaN. NumPy 2.5.2, default error settings. Each timing is the median of 7 batches after one warm-up (30 short reads, 3 whole-file reads, or 5 in-RAM reads per batch). "First open" is a single timing in a fresh process.

## Decisions and assumptions

- The lookup raises none of NumPy's floating-point warnings. Values are bit-identical to division; only the warnings differ. `open` already warns when a track's values exceed the output dtype.
- Tables are cached per process, keyed by (scale, dtype). Scales are powers of two, so a process sees few, and each table is at most 512 KB.

## Glossary

- **Division**: `np.divide(stored, scale)` on every read (main).
- **Guarded lookup**: table lookup only for reads ≥ 2^20 bins, with fallback to division whenever NumPy's error settings could report something for the window (previous branch head).
- **Cached lookup**: table lookup on every float16 read, table cached per process (this version).
