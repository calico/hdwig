# hdwig

Genomic coverage tracks in HDF5. One file per track, one 1-D array per contig,
float16 by default. It is an archival store — where a collection of tracks
lives between the BigWigs it came from and whatever consumes it — so
whole-genome loads are fast and random windows are fast too.

```
pip install hdwig
```

## Use

```python
import hdwig

track = hdwig.open("liver.hw")
track.contigs                        # {'chr1': 248956422, ...}
track.units, track.resolution, track.max
track.stats                          # Stats(max, sum, nonzero, nan, inf)
track["chr1", 1_000_000:1_131_072]   # true values, float16
track.read("chr1", 0, 1000, dtype="float32")
track.load("chr1")

hdwig.write("out.hw", {"chr1": coverage}, units="fragments/base/sample")
hdwig.merge("sum.hw", ["rep1.hw", "rep2.hw"], stat="sum")
```

```sh
hdwig convert in.bigwig out.hw -z --units 'reads/base'
hdwig convert a.bigwig,b.bigwig out.hw -m 0.5   # base-by-base sum, times 0.5
hdwig convert in.bedgraph out.hw -g genome.txt
hdwig convert old.w5 out.hw                     # rescale and recompress
hdwig merge -s sum out.hw rep1.hw rep2.hw
hdwig export in.hw out.bigwig
hdwig info in.hw
```

`write`, `merge` and `convert` all take the storage options — the fields of the
file being written: `--units`, `--resolution`, `--dtype`, `--chunk`,
`--compression` (`zstd`, `gzip`, `lzf`, `none`), `--level`, `--no-shuffle`,
`--headroom`. `convert` adds the few input transforms that have to happen
inside the streaming loop, before the genome is ever materialized: `-m`, `-z`,
`-i`.

Merge requires equal resolutions and equal lengths for shared contigs. It
inherits resolution and common units; differing units require an explicit
`units`/`--units` override. Missing contigs contribute zeros, with a warning.
An output resolution override must match the inputs.

## The format

An HDF5 file holding one 1-D dataset per contig, named by contig, dtype float16,
chunked at 2^16 values, zstd-3 with the byte shuffle filter. Root attributes:

| attr | meaning |
|---|---|
| `hdwig` | format version, currently 1 |
| `scale` | stored = true x scale (see below) |
| `max` | the largest absolute true value in the file |
| `sum`, `nonzero`, `nan`, `inf` | the rest of the summary |
| `resolution` | bp per value, default 1 |
| `units` | free text, e.g. `fragments/base/sample` |

Nucleotide resolution is the default. `resolution` records bp per bin; setting
it does not resample data. Reads and contig lengths use array indices (bins).
Exports multiply these coordinates by resolution to produce base-pair intervals.

The summary describes the values handed to `write`, before float16 rounded them,
and is recorded during writing. `hdwig info` reads this metadata without scanning
the data; `--scan` measures the stored values for comparison.

## Scale

Positive normal float16 values span about 6.1e-5 to 65504. Coverage tracks vary
over orders of magnitude between assays and depths, so a fixed encoding either saturates the deep tracks
or quantizes the shallow ones away. Each file therefore carries a **scale**: a
power of two chosen so the data sits high in float16's range with 4x headroom
left over.

```python
hdwig.scale_for(vmax)   # 2 ** floor(log2(65504 / (4 * vmax)))
```

Writers pick the scale; readers divide it out. Storage rounds values to float16
precision. Dividing by a power of two adds no rounding when the result stays
within the output dtype's normal range.

Two limits to consider:

- True values above 65504 cannot be held in float16 at all. `open()` sees this
  in the `max` attribute, without touching the data, and reads as float32
  instead, warning that it did. Pass `dtype` to override.
- Nonzero magnitudes below about 6.1e-5 lose precision or underflow on float16
  reads. Request float32 to retain the precision available in the stored values.

`write` refuses to store a finite value that would exceed the storage dtype's
ceiling, which is the failure the scale exists to prevent. `nan` and `inf` are stored as they came
and are counted but excluded from `max` and `sum`, so neither can move the scale:
`nan` means "no data", which is not zero and is the caller's to
interpret, and rewriting `inf` would only hide it.

## Legacy `.w5`

Files written before the format was named have no `hdwig` attribute. They read
through the same interface, taking `scale` from `w5_scale` if present and
carrying no summary; `hdwig info --scan` measures one. Values at the storage
ceiling suggest possible clipping, but do not prove it. `hdwig convert old.w5
new.hw` rewrites true values under the current spec, subject to the requested
storage precision. It warns about possible clipping; recovering clipped values
requires the original source.
