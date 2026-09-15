"""Conversion between hdwig and the interchange formats."""

import gzip
import warnings
from contextlib import ExitStack

import numpy as np

import hdwig


def _open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _interp_nan(x):
    """Fill NaN by linear interpolation, clamped at the ends."""
    nan = np.isnan(x)
    if not nan.any():
        return x
    if nan.all():
        return np.zeros_like(x)
    known = np.flatnonzero(~nan)
    x[nan] = np.interp(np.flatnonzero(nan), known, x[known])
    return x


def _runs(x):
    """(starts, ends, values) of the maximal constant runs of `x`."""
    change = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(x)]])
    return starts, ends, x[starts]


def _interval(values):
    """Which runs get an interval. Neither format has a way to say nan or inf,
    and an absent interval already reads back as nan, so only finite nonzeros
    are written."""
    return (values != 0) & np.isfinite(values)


################################################################################
# in
################################################################################
def from_bigwig(inputs, path, multiply=1.0, clip_neg=False, interp_nan=False, **kwargs):
    """Convert one BigWig, or the base-by-base sum of several, to `path`.

    The contig set comes from the first input. Headers give the maxima, so the
    conversion streams one contig at a time.
    """
    import pyBigWig

    with ExitStack() as stack:
        bws = []
        for f in inputs:
            bw = pyBigWig.open(str(f))
            stack.callback(bw.close)
            bws.append(bw)
        lengths = bws[0].chroms()
        for bw, name in zip(bws[1:], inputs[1:]):
            missing = set(lengths) - set(bw.chroms())
            if missing:
                raise ValueError(f"{name} is missing {sorted(missing)}")
        kwargs.setdefault("vmax", abs(multiply) * sum(bw.header()["maxVal"] for bw in bws))

        def contigs():
            for contig in sorted(lengths):
                length = lengths[contig]
                x = bws[0].values(contig, 0, length, numpy=True)
                for bw in bws[1:]:
                    x += bw.values(contig, 0, length, numpy=True)
                if multiply != 1:
                    x *= multiply
                x = _interp_nan(x) if interp_nan else np.nan_to_num(x)
                if clip_neg:
                    np.clip(x, 0, None, out=x)
                yield contig, x

        hdwig.write(path, contigs(), **kwargs)


def from_bedgraph(bedgraph, genome, path, **kwargs):
    """Convert a bedGraph to `path`, sizing contigs from a genome file.

    bedGraph is unindexed, so this holds the whole genome in float32 -- about
    12 GB for human -- and releases each contig as it is written. Entries are
    assumed not to overlap; where they do, the last one wins.
    """
    values = {}
    for line in _open_text(genome):
        contig, length = line.split()[:2]
        values[contig] = np.zeros(int(length), dtype="float32")

    with _open_text(bedgraph) as lines:
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or line.startswith(("#", "track", "browser")):
                continue
            contig, start, end, value = fields[0], int(fields[1]), int(fields[2]), float(fields[3])
            values[contig][start:end] = value

    kwargs.setdefault("vmax", max((float(np.abs(x).max()) for x in values.values() if len(x)), default=0.0))

    def drained():
        for contig in list(values):
            yield contig, values.pop(contig)

    hdwig.write(path, drained(), **kwargs)


def from_track(input, path, **kwargs):
    """Rewrite a track, most usefully a legacy `.w5`, under the current spec.

    Reads true values and writes them with the requested storage precision.
    Values at the source ceiling trigger a warning about possible clipping.
    """
    with hdwig.open(input, dtype="float32") as track:
        vmax = track.measure().max
        if vmax >= track.ceiling:
            warnings.warn(f"{input} may be clipped at {track.ceiling:g}; check its source")
        kwargs.setdefault("vmax", vmax)
        kwargs.setdefault("units", track.units)
        kwargs.setdefault("resolution", track.resolution)
        hdwig.write(path, ((c, track.read(c)) for c in track.contigs), **kwargs)


################################################################################
# out
################################################################################
def to_bigwig(path, out, contigs=None):
    """Write true values to a BigWig, one interval per constant run."""
    import pyBigWig

    with hdwig.open(path, dtype="float32") as track:
        contigs = contigs or list(track.contigs)
        bw = pyBigWig.open(str(out), "w")
        try:
            bw.addHeader([(c, track.contigs[c] * track.resolution) for c in contigs])
            for contig in contigs:
                starts, ends, values = _runs(track.read(contig))
                keep = _interval(values)
                if keep.any():
                    bw.addEntries(
                        [contig] * int(keep.sum()),
                        (starts[keep] * track.resolution).tolist(),
                        ends=(ends[keep] * track.resolution).tolist(),
                        values=values[keep].tolist(),
                    )
        finally:
            bw.close()


def to_bedgraph(path, out, contigs=None):
    """Write true values to a bedGraph, one line per nonzero constant run."""
    with hdwig.open(path, dtype="float32") as track, open(out, "w") as bg:
        for contig in contigs or track.contigs:
            starts, ends, values = _runs(track.read(contig))
            keep = _interval(values)
            for start, end, value in zip(*[a[keep] for a in (starts, ends, values)]):
                print(f"{contig}\t{start * track.resolution}\t{end * track.resolution}\t{value:.9g}", file=bg)
