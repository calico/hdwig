"""hdwig: genomic coverage tracks in HDF5.

One file per track, one 1-D array per contig, float16 by default. A per-track
power-of-two `scale` keeps the values inside float16's range; reads divide it
back out, so callers only ever see true values.
"""

import warnings
from contextlib import ExitStack
from typing import NamedTuple

import h5py
import hdf5plugin
import numpy as np

__version__ = "0.1.0"

FORMAT = 1  # value of the `hdwig` root attribute
EXT = ".hw"
CHUNK = 2**16
HEADROOM = 4

__all__ = ["Track", "Stats", "open", "write", "merge", "scale_for", "EXT", "FORMAT", "STATS"]


################################################################################
# summary
################################################################################
class Stats(NamedTuple):
    """What one pass over the data records, in true values.

    The writer makes this pass anyway, to pick the scale, so every file carries
    its own summary and `info` needs no scan. `nan` and `inf` are stored as they
    came but kept out of `max` and `sum`, so neither can move the scale or the
    mean.
    """

    max: float = 0.0
    sum: float = 0.0
    nonzero: int = 0
    nan: int = 0
    inf: int = 0

    def __add__(self, other):
        top = self.max if self.max > other.max else other.max
        return Stats(top, self.sum + other.sum, self.nonzero + other.nonzero,
                     self.nan + other.nan, self.inf + other.inf)


def _stats(x):
    """Summarize one contig without copying it; these arrays reach a gigabyte."""
    x = np.asarray(x, dtype="float32")
    finite = np.isfinite(x)
    nan = int(np.isnan(x).sum())
    return Stats(
        max=max(float(np.max(x, where=finite, initial=0.0)),
                -float(np.min(x, where=finite, initial=0.0))),
        sum=float(np.sum(x, where=finite, dtype="float64")),
        nonzero=int(np.count_nonzero(finite & (x != 0))),
        nan=nan,
        inf=len(x) - int(finite.sum()) - nan,
    )


################################################################################
# scale
################################################################################
def scale_for(vmax, headroom=HEADROOM, dtype="float16"):
    """The power-of-two storage factor for data topping out at `vmax`.

    Chosen to leave `headroom`x room under the dtype's ceiling, so that a later
    merge, or a higher peak in data not yet seen, still fits. A power of two
    makes the read-time divide exact within the output dtype's normal range.
    Only float16 needs one.
    """
    dtype = np.dtype(dtype)
    if dtype != np.float16 or not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    ceiling = float(np.finfo(dtype).max)
    return float(2.0 ** np.floor(np.log2(ceiling / (headroom * vmax))))


################################################################################
# read
################################################################################
class Track:
    """A coverage track, open for reading.

    Reads return true values in the stored dtype, unless `dtype` asks for
    another. Legacy `.w5` files, which carry `w5_scale` instead of `scale` and
    no `max`, read through the same interface.
    """

    def __init__(self, path, dtype=None):
        self.path = str(path)
        self.h5 = h5py.File(self.path, "r")
        attrs = self.h5.attrs
        self.format = int(attrs.get("hdwig", 0))  # 0: legacy .w5
        self.scale = float(attrs.get("scale", attrs.get("w5_scale", 1.0)))
        self.resolution = int(attrs.get("resolution", 1))
        self.units = str(attrs.get("units", ""))
        self.max = float(attrs["max"]) if "max" in attrs else None
        self.contigs = {name: dset.shape[0] for name, dset in self.h5.items()}
        # legacy .w5 records none of this; `measure()` gets it by reading
        self.stats = Stats(
            self.max, float(attrs["sum"]), int(attrs["nonzero"]),
            int(attrs["nan"]), int(attrs["inf"]),
        ) if "sum" in attrs else None

        dsets = list(self.h5.values())
        self.stored_dtype = dsets[0].dtype if dsets else np.dtype("float16")
        self.dtype = np.dtype(dtype) if dtype is not None else self.stored_dtype

        # a track can declare values the stored dtype cannot represent once the
        # scale is divided out; rather than clip, widen, and say so
        ceiling = float(np.finfo(self.dtype).max)
        if self.max is not None and self.max > ceiling:
            fix = "reading as float32" if dtype is None else "pass dtype='float32'"
            warnings.warn(
                f"{self.path}: values reach {self.max:g}, above {self.dtype}'s "
                f"{ceiling:g}; {fix}.",
                stacklevel=2,
            )
            if dtype is None:
                self.dtype = np.dtype("float32")

    def read(self, contig, start=None, end=None, dtype=None):
        """True values over `contig[start:end]`."""
        x = self.h5[contig][start:end]
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        if self.scale == 1:
            return x.astype(dtype, copy=False)
        return np.divide(x, self.scale, dtype=dtype)

    def load(self, contig, dtype=None):
        """True values over the whole contig."""
        return self.read(contig, dtype=dtype)

    def measure(self):
        """`Stats` from the data, for a legacy file that records none, or to
        check the ones a file claims."""
        total = Stats()
        for contig in self.contigs:
            total += _stats(self.read(contig, dtype="float32"))
        return total

    @property
    def ceiling(self):
        """The largest true value this file can hold; reaching it may indicate clipping."""
        return float(np.finfo(self.stored_dtype).max) / self.scale

    def __getitem__(self, key):
        contig, window = key if isinstance(key, tuple) else (key, slice(None))
        if window.step not in (None, 1):
            raise ValueError("slice step must be 1")
        return self.read(contig, window.start, window.stop)

    def __contains__(self, contig):
        return contig in self.contigs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"<hdwig.Track {self.path} {len(self.contigs)} contigs scale={self.scale:g}>"

    def close(self):
        self.h5.close()


def open(path, dtype=None):
    """Open a track for reading. Also reads legacy `.w5`."""
    return Track(path, dtype=dtype)


################################################################################
# write
################################################################################
def _filters(compression, level, shuffle):
    if compression in (None, "none"):
        return {}
    if compression == "zstd":
        return dict(shuffle=shuffle, **hdf5plugin.Zstd(clevel=level))
    if compression == "gzip":
        return dict(shuffle=shuffle, compression="gzip", compression_opts=level)
    if compression == "lzf":
        return dict(shuffle=shuffle, compression="lzf")
    raise ValueError(f"unknown compression {compression!r}; use zstd, gzip, lzf or none")


def write(
    path,
    arrays,
    units="",
    resolution=1,
    vmax=None,
    headroom=HEADROOM,
    dtype="float16",
    chunk=CHUNK,
    compression="zstd",
    level=3,
    shuffle=True,
):
    """Write true-valued coverage to `path`.

    `arrays` maps contig name to a 1-D array, or is any iterable of
    (name, array) pairs. Pass `vmax`, the largest absolute true value, to
    stream those pairs; without it they are materialized so it can be measured.
    An overstated `vmax` only costs headroom, an understated one raises.
    """
    items = arrays.items() if hasattr(arrays, "items") else arrays
    if vmax is None:
        arrays = dict(items)
        vmax = max((_stats(x).max for x in arrays.values()), default=0.0)
        items = arrays.items()

    scale = scale_for(vmax, headroom, dtype)
    dtype = np.dtype(dtype)
    ceiling = float(np.finfo(dtype).max)
    opts = _filters(compression, level, shuffle)
    total = Stats()

    with h5py.File(path, "w") as h5:
        h5.attrs["hdwig"] = FORMAT
        h5.attrs["scale"] = scale
        h5.attrs["resolution"] = int(resolution)
        h5.attrs["units"] = units

        for contig, x in items:
            x = np.array(x, dtype="float32")
            stats = _stats(x)
            total += stats
            # checked before scaling, so a value the scale sends to inf cannot
            # slip past as one more non-finite
            if stats.max * scale > ceiling:
                raise ValueError(
                    f"{contig} reaches {stats.max:g}, above what {dtype} holds at "
                    f"scale {scale:g}. Pass a vmax of at least {stats.max:g}."
                )
            # nan and inf pass through. nan means "no data", which is not zero and
            # is the caller's to interpret; inf is corruption, but inventing a
            # finite value for it would hide that. Neither reaches the scale.
            if scale != 1:
                x *= scale
            h5.create_dataset(
                contig, data=x.astype(dtype), chunks=(max(1, min(chunk, len(x))),), **opts
            )

        # measured, not inferred from the vmax the scale came from
        for field, value in total._asdict().items():
            h5.attrs[field] = value


################################################################################
# merge
################################################################################
STATS = {
    "sum": lambda a: a.sum(axis=0),
    "mean": lambda a: a.mean(axis=0),
    "geo-mean": lambda a: np.exp(np.log(a).mean(axis=0)),
    "sqrt-mean": lambda a: np.sqrt(a).mean(axis=0) ** 2,
}


def merge(path, inputs, stat="sum", **kwargs):
    """Combine tracks base by base in true values, contigs unioned.

    Inputs must agree on resolution and shared contig lengths. Common units
    and resolution are inherited; differing units require an explicit override.
    Every statistic is monotone in each input, so applying it to the inputs'
    maxima bounds the output's -- enough to pick the scale and stream.
    """
    with ExitStack() as stack:
        tracks = [stack.enter_context(open(p, dtype="float32")) for p in inputs]
        if not tracks:
            raise ValueError("merge requires at least one input")
        resolution = tracks[0].resolution
        if any(t.resolution != resolution for t in tracks):
            raise ValueError("input resolutions must match")
        if kwargs.setdefault("resolution", resolution) != resolution:
            raise ValueError("output resolution must match input resolution")
        if "units" not in kwargs:
            if any(t.units != tracks[0].units for t in tracks):
                raise ValueError("input units differ; pass units explicitly")
            kwargs["units"] = tracks[0].units

        contigs = {}
        for track in tracks:
            for contig, length in track.contigs.items():
                if contigs.setdefault(contig, length) != length:
                    raise ValueError(f"{contig}: input contig lengths must match")

        maxes = np.array([[t.max if t.max is not None else t.measure().max] for t in tracks], "float32")
        kwargs.setdefault("vmax", float(STATS[stat](maxes)[0]))

        def summarized():
            for contig, length in contigs.items():
                stacked = np.zeros((len(tracks), length), dtype="float32")
                for i, track in enumerate(tracks):
                    if contig in track.contigs:
                        stacked[i] = track.read(contig)
                    else:
                        warnings.warn(f"{track.path} is missing {contig}", stacklevel=2)
                yield contig, STATS[stat](stacked)

        write(path, summarized(), **kwargs)
