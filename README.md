# hdwig

Genomic coverage tracks in HDF5. One file per track, one 1-D array per contig,
float16 by default. It is an archival store — where a collection of tracks
lives between the BigWigs it came from and whatever consumes it — so
whole-genome loads are fast and random windows are fast too.

Against the same seven tracks as BigWig, hdwig is 3.1x smaller and 2x faster to
read a 128 kb window; against a gzip-compressed HDF5, 2.9x faster to load and 5x
faster to write.

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
hdwig convert old.w5 out.hw                     # rescale and recompress in place
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

Nucleotide resolution is the default, not a requirement: `resolution` records
the bin width, and everything else is unchanged.

The summary describes the values handed to `write`, before float16 rounded them,
and costs nothing: the writer passes over the data anyway, to pick the scale.
Unlike BigWig, hdwig stores no zoom levels, so without it every statistic would
be a full read — 44 s for a human genome. `hdwig info` is therefore instant, and
`--scan` measures the stored values for comparison.

## Scale

float16 spans 6.1e-5 to 65504. Coverage tracks vary over orders of magnitude
between assays and depths, so a fixed encoding either saturates the deep tracks
or quantizes the shallow ones away. Each file therefore carries a **scale**: a
power of two chosen so the data sits high in float16's range with 4x headroom
left over.

```python
hdwig.scale_for(vmax)   # 2 ** floor(log2(65504 / (4 * vmax)))
```

**You do not have to think about this.** Writers pick the scale; readers divide
it out. Because it is a power of two, the divide is exponent arithmetic — exact,
not merely close — so reads hand back true values in float16 with nothing lost.

Two edges the exactness argument does not cover, both reported by `hdwig info`:

- True values above 65504 cannot be held in float16 at all. `open()` sees this
  in the `max` attribute, without touching the data, and reads as float32
  instead, warning that it did. Pass `dtype` to override.
- True values below 6.1e-5 land in float16's subnormals and lose precision.

`write` refuses to store a value that would reach the float16 ceiling, which is
the failure the scale exists to prevent. `nan` and `inf` are stored as they came
and are counted but excluded from `max` and `sum`, so neither can move the scale
or the mean: `nan` means "no data", which is not zero and is the caller's to
interpret, and rewriting `inf` would only hide it.

## Legacy `.w5`

Files written before the format was named have no `hdwig` attribute. They read
through the same interface, taking `scale` from `w5_scale` if present and
carrying no summary; `hdwig info --scan` measures one and reports whether the
file was clipped. `hdwig convert old.w5 new.hw` rewrites one under the current
spec, preserving true values exactly. Clipping already in a file cannot be
undone that way — that needs the original BigWig — so it warns instead.
