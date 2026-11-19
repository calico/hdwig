# When does a float16 lookup pay off?

Opening a track for 128 bases takes about 1.4 ms with direct division. Checking for exceptional inputs before using a lookup retains most of the speedup on a whole contig.

| Operation | Direct division | Unconditional lookup | Guarded lookup |
|---|---:|---:|---:|
| Open and read 128 bases | 1.40 ms | 3.11 ms | 1.39 ms |
| Open and read 4,194,304 bases | 47.6 ms | 28.8 ms | 30.1 ms |
| Read 4,194,304 bases already in RAM | 28.4 ms | 7.76 ms | 8.87 ms |

## Method

Each implementation read the same local zstd-compressed float16 track. Inputs were 4,194,304 exponentially distributed float32 values with mean 3, generated with NumPy seed 441 and the default writer settings. NumPy 2.4.6 used its default error policy. Times are medians of seven batches after warm-up, with 30 short reads, three whole-file reads or five in-memory reads per batch. Each in-memory track had already served one whole-contig read.

## Decisions and assumptions

Use direct division for first reads below 1,048,576 bins. The setup cost varies with scale. In an array-only check at 524,288 bins and scale 8192, building and checking the lookup took 3.65 ms versus 3.55 ms for division. The cutoff is a conservative choice for this machine.

For large finite float16 windows, use the lookup when the scale and error settings permit it. Restricting the lookup to callers who silence every error would give default callers the direct-division timings above. Building a fresh table from only the values present took about 21 ms for 4,194,304 in-memory values in the array-only check.
