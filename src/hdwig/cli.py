"""The `hdwig` command line."""

import argparse

import numpy as np

import hdwig
from hdwig import convert

BIGWIG = (".bw", ".bigwig")
BEDGRAPH = (".bedgraph", ".bg", ".bdg")
HDWIG = (".hw", ".w5", ".h5", ".hdf5", ".wdf5")


def _suffixed(path, exts):
    name = str(path).lower()
    name = name[:-3] if name.endswith(".gz") else name
    return name.endswith(exts)


def _storage(args):
    """The write options that were actually given; the rest keep their defaults,
    or, converting one track to another, are inherited from the source."""
    keys = ("units", "resolution", "headroom", "dtype", "chunk", "level", "shuffle")
    opts = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    opts["compression"] = None if args.compression == "none" else args.compression
    return opts


def _convert(args):
    inputs = args.input.split(",")
    if _suffixed(inputs[0], BIGWIG):
        convert.from_bigwig(
            inputs, args.output, multiply=args.multiply, clip_neg=args.clip_neg,
            interp_nan=args.interp_nan, **_storage(args),
        )
    elif _suffixed(inputs[0], BEDGRAPH):
        if args.genome is None:
            raise SystemExit("bedGraph input needs -g/--genome for the contig lengths")
        convert.from_bedgraph(inputs[0], args.genome, args.output, **_storage(args))
    elif _suffixed(inputs[0], HDWIG):
        convert.from_track(inputs[0], args.output, **_storage(args))
    else:
        raise SystemExit(f"cannot tell what {inputs[0]} is; expected a BigWig, bedGraph or .w5")


def _merge(args):
    hdwig.merge(args.output, args.inputs, stat=args.stat, **_storage(args))


def _export(args):
    contigs = args.contigs.split(",") if args.contigs else None
    write = convert.to_bigwig if _suffixed(args.output, BIGWIG) else convert.to_bedgraph
    write(args.input, args.output, contigs=contigs)


def _info(args):
    with hdwig.open(args.input) as track:
        bases = sum(track.contigs.values())
        stats = track.measure() if args.scan else track.stats

        print(track.path)
        print(f"  format      {track.format or 'legacy .w5'}")
        print(f"  units       {track.units or 'unspecified'}")
        print(f"  resolution  {track.resolution} bp")
        print(f"  dtype       {track.stored_dtype}")
        print(f"  scale       {track.scale:g}")
        print(f"  contigs     {len(track.contigs)}, {bases / 1e9:.2f} Gb")

        if stats is None:
            print("  statistics  unrecorded; --scan to measure them")
            return

        print(f"  max         {stats.max:g}")
        print(f"  mean        {stats.sum / bases:g}")
        print(f"  nonzero     {100 * stats.nonzero / bases:.2f}%")
        print(f"  nan         {stats.nan} ({100 * stats.nan / bases:.2f}%)")
        print(f"  inf         {stats.inf}")

        if stats.max >= track.ceiling:
            print(f"  SATURATED   values were clipped at {track.ceiling:g}; reconvert from the source")
        elif stats.max > float(np.finfo("float16").max):
            print("  read with dtype='float32'; the true values exceed float16's range")


def main(argv=None):
    storage = argparse.ArgumentParser(add_help=False)
    storage.add_argument("--chunk", type=int, default=hdwig.CHUNK, help="values per chunk [%(default)s]")
    storage.add_argument("--compression", default="zstd", choices=["zstd", "gzip", "lzf", "none"],
                         help="[%(default)s]")
    storage.add_argument("--dtype", default="float16", help="[%(default)s]")
    storage.add_argument("--headroom", type=float, default=hdwig.HEADROOM,
                         help="room to leave under the dtype ceiling [%(default)s]")
    storage.add_argument("--level", type=int, default=3, help="compression level [%(default)s]")
    storage.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                         help="skip the byte shuffle filter")
    storage.add_argument("--resolution", type=int, help="bp per value [1]")
    storage.add_argument("--units", help="e.g. 'fragments/base/sample'")

    parser = argparse.ArgumentParser(prog="hdwig", description=__doc__)
    parser.add_argument("--version", action="version", version=hdwig.__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", parents=[storage], help="BigWig or bedGraph in")
    p.add_argument("input", help="input file, or a comma-separated list of BigWigs to sum")
    p.add_argument("output")
    p.add_argument("-z", "--clip-neg", action="store_true", help="clip negative values at zero")
    p.add_argument("-g", "--genome", help="contig lengths; required for bedGraph input")
    p.add_argument("-i", "--interp-nan", action="store_true",
                   help="interpolate NaN rather than zeroing it")
    p.add_argument("-m", "--multiply", type=float, default=1.0,
                   help="multiply values, e.g. to normalize while summing [%(default)s]")
    p.set_defaults(run=_convert)

    p = sub.add_parser("merge", parents=[storage], help="combine tracks base by base")
    p.add_argument("output")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-s", "--stat", default="sum", choices=list(hdwig.STATS), help="[%(default)s]")
    p.set_defaults(run=_merge)

    p = sub.add_parser("export", help="BigWig or bedGraph out")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-c", "--contigs", help="comma-separated subset")
    p.set_defaults(run=_export)

    p = sub.add_parser("info", help="attributes and contigs")
    p.add_argument("input")
    p.add_argument("--scan", action="store_true",
                   help="measure the statistics from the stored values; the recorded ones "
                        "describe the input, before float16 rounded it")
    p.set_defaults(run=_info)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
