import h5py
import numpy as np
import pytest

import hdwig
from hdwig import cli, convert

F16MAX = float(np.finfo("float16").max)
SMALLEST_NORMAL = float(np.finfo("float16").smallest_normal)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {
        "chr1": rng.exponential(3.0, 5000).astype("float32"),
        "chr2": np.zeros(1000, dtype="float32"),
        "chrM": rng.exponential(50.0, 300).astype("float32"),
    }


def _handmade(path, arrays, **attrs):
    """A file with attributes set by hand, for the read path."""
    with h5py.File(path, "w") as h5:
        h5.attrs.update(attrs)
        for contig, x in arrays.items():
            h5.create_dataset(contig, data=np.asarray(x, "float16"))
    return path


################################################################################
# scale
################################################################################
def test_scale_leaves_headroom():
    for vmax in (1e-3, 1.0, 7.3, 1000.0, 60000.0):
        scale = hdwig.scale_for(vmax)
        assert np.log2(scale) == int(np.log2(scale))  # a power of two
        assert vmax * scale * hdwig.HEADROOM <= F16MAX
        assert vmax * scale * hdwig.HEADROOM > F16MAX / 2  # and the largest such


def test_scale_only_for_float16():
    assert hdwig.scale_for(1e-3, dtype="float32") == 1.0
    assert hdwig.scale_for(0.0) == 1.0


def test_scale_divide_is_exact(tmp_path):
    """The claim the automatic scale rests on: dividing a power of two out of
    float16 is exponent arithmetic, exact until the result leaves normal range."""
    stored = np.array([1, 3, 7, 1023, 2047, 60000, 0.5, 1e-4], "float16")
    for exponent in range(-4, 13):
        scale = 2.0**exponent
        track = hdwig.open(_handmade(tmp_path / f"s{exponent}.hw", {"c": stored},
                                     hdwig=1, scale=scale))
        true16, true32 = track.read("c"), track.read("c", dtype="float32")

        assert true32.dtype == np.float32 and true16.dtype == np.float16
        np.testing.assert_array_equal(true32 * scale, stored.astype("float32"))

        normal = (np.abs(true32) >= SMALLEST_NORMAL) & (np.abs(true32) <= F16MAX)
        np.testing.assert_array_equal(
            true16[normal].astype("float32") * scale, stored[normal].astype("float32")
        )
        track.close()


def test_scale_divide_loses_subnormals(tmp_path):
    """The one place it is not exact, and what to do about it."""
    stored = np.array([1e-4], "float16")
    track = hdwig.open(_handmade(tmp_path / "sub.hw", {"c": stored}, hdwig=1, scale=1024.0))
    assert track.read("c")[0] != track.read("c", dtype="float32")[0]
    assert track.read("c", dtype="float32")[0] * 1024.0 == stored.astype("float32")[0]


################################################################################
# write and read
################################################################################
def test_round_trip(tmp_path):
    stored = {"c": np.array([0.0, 1.0, 2.0, 4.0, 8.0], "float32")}
    path = tmp_path / "t.hw"
    hdwig.write(path, stored, units="reads/base", resolution=1)
    with hdwig.open(path) as track:
        assert track.format == hdwig.FORMAT
        assert track.units == "reads/base"
        assert track.contigs == {"c": 5}
        assert track.max == 8.0
        assert track.scale == 2.0 ** np.floor(np.log2(F16MAX / 32))
        np.testing.assert_array_equal(track.read("c", dtype="float32"), stored["c"])


def test_values_survive(tmp_path, arrays):
    path = tmp_path / "t.hw"
    hdwig.write(path, arrays)
    with hdwig.open(path) as track:
        for contig, x in arrays.items():
            got = track.read(contig, dtype="float32")
            assert got.shape == x.shape
            np.testing.assert_allclose(got, x, rtol=1e-3)  # float16 is ~5e-4


def test_slicing(tmp_path, arrays):
    path = tmp_path / "t.hw"
    hdwig.write(path, arrays)
    with hdwig.open(path) as track:
        whole = track.load("chr1")
        np.testing.assert_array_equal(track["chr1", 100:200], whole[100:200])
        np.testing.assert_array_equal(track["chr1"], whole)
        np.testing.assert_array_equal(track.read("chr1", 100), whole[100:])


def test_nonfinite_survives_the_round_trip(tmp_path):
    """Real tracks carry nan, and a few carry stray inf. nan is "no data", which
    downstream fills with a baseline rather than zero, so the writer must not
    decide it means zero; inf is corruption, and hiding it would be worse than
    passing it on. Neither may reach the scale."""
    path = tmp_path / "t.hw"
    x = np.array([np.nan, 4.0, np.inf, -np.inf], "float32")
    hdwig.write(path, {"c": x})
    with hdwig.open(path) as track:
        assert track.scale == hdwig.scale_for(4.0)  # 4.0 set it, not inf
        assert track.max == 4.0
        assert track.stats.nan == 1 and track.stats.inf == 2
        np.testing.assert_array_equal(track.read("c", dtype="float32"), x)


def test_statistics_recorded(tmp_path):
    """The writer passes over the data anyway, to pick the scale, so it records
    the summary and `info` needs no scan."""
    x = np.array([0.0, -1.0, 3.0, np.nan, np.inf], "float32")
    path = tmp_path / "t.hw"
    hdwig.write(path, {"a": x, "b": np.zeros(4, "float32")})
    with hdwig.open(path) as track:
        assert track.stats == hdwig.Stats(max=3.0, sum=2.0, nonzero=2, nan=1, inf=1)
        assert track.stats.max == track.max


def test_understated_vmax_raises(tmp_path):
    with pytest.raises(ValueError, match="above what"):
        hdwig.write(tmp_path / "t.hw", {"c": np.array([1.0, 1e6], "float32")}, vmax=1.0)


def test_understated_vmax_raises_even_past_float32(tmp_path):
    """The scale can send a finite value past float32's own range; checking
    before the multiply keeps it from passing as one more inf."""
    with pytest.raises(ValueError, match="above what"):
        hdwig.write(tmp_path / "t.hw", {"c": np.array([1e35], "float32")}, vmax=1.0)


def test_beyond_float16_warns_and_float32_recovers(tmp_path):
    """A track whose true values do not fit float16: the scale still stores it,
    and open() widens to float32 on the strength of the `max` attr alone, with
    no data read. Asking for float16 anyway is allowed, and warns."""
    path = tmp_path / "big.hw"
    x = np.array([0.0, 1e5, 2e5], "float32")
    hdwig.write(path, {"c": x})

    with pytest.warns(UserWarning, match="reading as float32"):
        track = hdwig.open(path)
    assert track.dtype == np.dtype("float32")
    np.testing.assert_allclose(track.read("c"), x, rtol=1e-3)
    track.close()

    with pytest.warns(UserWarning, match="pass dtype='float32'"):
        track = hdwig.open(path, dtype="float16")
    assert np.isinf(track.read("c")[2])  # which is why the warning is there
    track.close()


@pytest.mark.parametrize("compression", ["zstd", "gzip", "lzf", "none"])
def test_compression_options(tmp_path, arrays, compression):
    path = tmp_path / f"{compression}.hw"
    hdwig.write(path, arrays, compression=compression, chunk=1024)
    with hdwig.open(path) as track:
        assert track.h5["chr1"].chunks == (1024,)
        np.testing.assert_allclose(track.read("chr1", dtype="float32"), arrays["chr1"], rtol=1e-3)


def test_float32_storage(tmp_path, arrays):
    path = tmp_path / "f32.hw"
    hdwig.write(path, arrays, dtype="float32")
    with hdwig.open(path) as track:
        assert track.scale == 1.0 and track.stored_dtype == np.float32
        np.testing.assert_array_equal(track.read("chr1"), arrays["chr1"])


def test_streaming_write(tmp_path, arrays):
    """Pairs plus a vmax, so nothing is materialized."""
    path = tmp_path / "t.hw"
    hdwig.write(path, iter(arrays.items()), vmax=max(x.max() for x in arrays.values()))
    with hdwig.open(path) as track:
        assert set(track.contigs) == set(arrays)


################################################################################
# legacy .w5
################################################################################
def test_legacy_w5(tmp_path):
    stored = np.array([1.0, 2.0, 400.0], "float16")
    path = _handmade(tmp_path / "old.w5", {"chr1": stored}, w5_scale=0.25,
                     units="fragments/base/sample")
    with hdwig.open(path) as track:
        assert track.format == 0 and track.scale == 0.25 and track.max is None
        np.testing.assert_array_equal(track.read("chr1", dtype="float32"), stored * 4)
        assert track.stats is None  # a legacy file records none; reading gets them
        assert track.measure() == hdwig.Stats(max=1600.0, sum=1612.0, nonzero=3)


def test_legacy_w5_without_attrs(tmp_path):
    path = _handmade(tmp_path / "bare.w5", {"chr1": np.arange(10, dtype="float16")})
    with hdwig.open(path) as track:
        assert track.scale == 1.0 and track.units == ""
        np.testing.assert_array_equal(track.read("chr1"), np.arange(10))


def test_saturation_detected(tmp_path):
    clipped = _handmade(tmp_path / "clip.w5", {"chr1": np.array([1.0, F16MAX], "float16")})
    with hdwig.open(clipped) as track:
        assert track.ceiling == F16MAX and track.measure().max >= track.ceiling
    scaled = _handmade(tmp_path / "ok.w5", {"chr1": np.array([1.0, 2.0], "float16")}, w5_scale=4.0)
    with hdwig.open(scaled) as track:
        assert track.ceiling == F16MAX / 4 and track.measure().max < track.ceiling


def test_from_track(tmp_path):
    """Rewriting a legacy .w5 keeps true values and picks a new scale."""
    stored = np.array([1.0, 2.0, 400.0], "float16")
    old = _handmade(tmp_path / "old.w5", {"chr1": stored}, w5_scale=0.25, units="reads/base")
    new = tmp_path / "new.hw"
    convert.from_track(old, new)
    with hdwig.open(new) as track:
        assert track.format == hdwig.FORMAT and track.units == "reads/base"
        assert track.scale == hdwig.scale_for(1600.0) and track.max == 1600.0
        np.testing.assert_array_equal(track.read("chr1", dtype="float32"), stored * 4)


def test_from_track_warns_when_clipped(tmp_path):
    old = _handmade(tmp_path / "clip.w5", {"chr1": np.array([1.0, F16MAX], "float16")})
    with pytest.warns(UserWarning, match="may be clipped"):
        convert.from_track(old, tmp_path / "new.hw")


################################################################################
# merge
################################################################################
@pytest.mark.parametrize("stat,expected", [("sum", 9.0), ("mean", 3.0), ("sqrt-mean", 2.7427)])
def test_merge(tmp_path, stat, expected):
    paths = []
    for i, value in enumerate([1.0, 3.0, 5.0]):
        paths.append(tmp_path / f"in{i}.hw")
        hdwig.write(paths[-1], {"chr1": np.full(100, value, "float32")})
    out = tmp_path / "merged.hw"
    hdwig.merge(out, paths, stat=stat)
    with hdwig.open(out) as track:
        np.testing.assert_allclose(track.read("chr1", dtype="float32"), expected, rtol=1e-3)


def test_merge_unions_contigs(tmp_path):
    a, b = tmp_path / "a.hw", tmp_path / "b.hw"
    hdwig.write(a, {"chr1": np.ones(10, "float32")})
    hdwig.write(b, {"chr2": np.ones(20, "float32")})
    out = tmp_path / "m.hw"
    with pytest.warns(UserWarning, match="missing"):
        hdwig.merge(out, [a, b])
    with hdwig.open(out) as track:
        assert track.contigs == {"chr1": 10, "chr2": 20}


################################################################################
# conversion
################################################################################
@pytest.fixture
def pyBigWig():
    return pytest.importorskip("pyBigWig")


@pytest.fixture
def bigwig(tmp_path, pyBigWig):
    path = str(tmp_path / "in.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("chr1", 1000)])
    bw.addEntries(["chr1"] * 3, [10, 100, 500], ends=[20, 200, 600], values=[1.0, 4.0, 16.0])
    bw.close()
    return path


def test_from_bigwig(tmp_path, bigwig):
    out = tmp_path / "t.hw"
    convert.from_bigwig([bigwig], out)
    with hdwig.open(out) as track:
        x = track.read("chr1", dtype="float32")
        assert track.contigs == {"chr1": 1000}
        assert (x[10:20] == 1.0).all() and (x[100:200] == 4.0).all() and (x[500:600] == 16.0).all()
        assert x[0] == 0.0 and track.max == 16.0


def test_from_bigwig_sums_and_multiplies(tmp_path, bigwig):
    out = tmp_path / "t.hw"
    convert.from_bigwig([bigwig, bigwig], out, multiply=0.5)
    with hdwig.open(out) as track:
        np.testing.assert_array_equal(track.read("chr1", dtype="float32")[10:20], 1.0)


def test_from_bigwig_options(tmp_path, bigwig):
    out = tmp_path / "t.hw"
    convert.from_bigwig([bigwig], out, multiply=-1.0, clip_neg=True)
    with hdwig.open(out) as track:
        assert (track.read("chr1", dtype="float32") == 0.0).all()


def test_bigwig_round_trip(tmp_path, bigwig, pyBigWig):
    hw, out = tmp_path / "t.hw", str(tmp_path / "out.bw")
    convert.from_bigwig([bigwig], hw)
    convert.to_bigwig(hw, out)
    bw = pyBigWig.open(out)
    x = np.nan_to_num(bw.values("chr1", 0, 1000, numpy=True))
    bw.close()
    with hdwig.open(hw) as track:
        np.testing.assert_array_equal(x, track.read("chr1", dtype="float32"))


def test_export_skips_nonfinite(tmp_path, pyBigWig):
    """A track holding nan exports without one interval per nan base; BigWig
    says "no data" by leaving the interval out, which reads back as nan."""
    hw, out = tmp_path / "t.hw", str(tmp_path / "out.bw")
    x = np.array([1.0] * 5 + [np.nan] * 5, "float32")
    hdwig.write(hw, {"chr1": x})
    convert.to_bigwig(hw, out)
    bw = pyBigWig.open(out)
    assert bw.intervals("chr1") == ((0, 5, 1.0),)
    np.testing.assert_array_equal(bw.values("chr1", 0, 10, numpy=True), x)
    bw.close()


def test_bedgraph(tmp_path):
    genome, bg = tmp_path / "genome.txt", tmp_path / "in.bedgraph"
    genome.write_text("chr1\t100\n")
    bg.write_text("track type=bedGraph\nchr1\t0\t10\t2.0\nchr1\t5\t20\t4.0\n")

    out = tmp_path / "flat.hw"
    convert.from_bedgraph(bg, genome, out)
    with hdwig.open(out) as track:
        x = track.read("chr1", dtype="float32")
        # the two entries overlap over 5:10, where the later one wins
        assert x[0] == 2.0 and x[7] == 4.0 and x[50] == 0.0


def test_to_bedgraph(tmp_path):
    hw, bg = tmp_path / "t.hw", tmp_path / "out.bedgraph"
    x = np.zeros(20, "float32")
    x[5:10] = 3.0
    hdwig.write(hw, {"chr1": x})
    convert.to_bedgraph(hw, bg)
    assert bg.read_text() == "chr1\t5\t10\t3\n"


################################################################################
# cli
################################################################################
def test_cli(tmp_path, bigwig, capsys):
    hw = str(tmp_path / "t.hw")
    cli.main(["convert", bigwig, hw, "-z", "--units", "reads/base"])
    with hdwig.open(hw) as track:
        assert track.units == "reads/base"

    merged = str(tmp_path / "m.hw")
    cli.main(["merge", merged, hw, hw, "-s", "sum"])
    with hdwig.open(merged) as track:
        assert track.max == 32.0

    cli.main(["export", merged, str(tmp_path / "out.bw")])
    assert (tmp_path / "out.bw").exists()

    cli.main(["info", merged])
    out = capsys.readouterr().out
    assert "max         32" in out and "contigs     1" in out

    rewritten = str(tmp_path / "r.hw")
    cli.main(["convert", hw, rewritten, "--compression", "gzip"])
    with hdwig.open(rewritten) as track:
        assert track.units == "reads/base"  # inherited, not blanked


def test_cli_info_saturated(tmp_path, capsys):
    path = _handmade(tmp_path / "clip.w5", {"chr1": np.array([1.0, F16MAX], "float16")})
    cli.main(["info", str(path), "--scan"])
    out = capsys.readouterr().out
    assert "legacy .w5" in out and "possible clipping" in out


def test_cli_bedgraph_needs_genome(tmp_path):
    with pytest.raises(SystemExit, match="genome"):
        cli.main(["convert", str(tmp_path / "x.bedgraph"), str(tmp_path / "y.hw")])


@pytest.mark.parametrize("step", [0, 2, -1])
def test_slicing_rejects_steps(tmp_path, step):
    path = tmp_path / "t.hw"
    hdwig.write(path, {"c": np.arange(4, dtype="float32")})
    with hdwig.open(path) as track:
        with pytest.raises(ValueError, match="slice step"):
            track["c", ::step]
        np.testing.assert_array_equal(track["c", ::1], track.read("c"))


@pytest.mark.parametrize("values,expected", [
    ([1, np.inf, -np.inf], [1, np.inf, -np.inf]),
    ([np.nan, np.nan], [0, 0]),
    ([np.nan, 1, np.nan, 3, np.nan], [1, 1, 2, 3, 3]),
    ([np.inf, 1, np.nan, 3, -np.inf], [np.inf, 1, 2, 3, -np.inf]),
])
def test_interpolate_nan(values, expected):
    np.testing.assert_array_equal(convert._interp_nan(np.array(values, "float32")), expected)


@pytest.mark.parametrize("case,message", [
    ("empty", "at least one"),
    ("length", "lengths must match"),
    ("resolution", "resolutions must match"),
    ("override", "output resolution"),
    ("units", "units differ"),
])
def test_merge_rejects_incompatible_inputs_before_writing(tmp_path, case, message):
    a, b, out = (tmp_path / name for name in ("a.hw", "b.hw", "out.hw"))
    hdwig.write(a, {"c": np.ones(4)}, resolution=10, units="reads")
    hdwig.write(b, {"c": np.ones(1 if case == "length" else 4)},
                resolution=1 if case == "resolution" else 10,
                units="other" if case == "units" else "reads")
    out.write_bytes(b"existing output")
    kwargs = {"resolution": 1} if case == "override" else {}
    with pytest.raises(ValueError, match=message):
        hdwig.merge(out, [] if case == "empty" else [a, b], **kwargs)
    assert out.read_bytes() == b"existing output"


def test_merge_metadata(tmp_path):
    a, b, out = (tmp_path / name for name in ("a.hw", "b.hw", "out.hw"))
    hdwig.write(a, {"c": np.ones(4)}, resolution=10, units="reads")
    hdwig.merge(out, [a, a])
    with hdwig.open(out) as track:
        assert (track.resolution, track.units) == (10, "reads")
    hdwig.write(b, {"c": np.ones(4)}, resolution=10, units="other")
    hdwig.merge(out, [a, b], units="combined", resolution=10)
    with hdwig.open(out) as track:
        assert (track.resolution, track.units) == (10, "combined")


def test_merge_closes_inputs_on_open_failure(tmp_path, monkeypatch):
    path, out = tmp_path / "a.hw", tmp_path / "out.hw"
    hdwig.write(path, {"c": np.ones(4)})
    track = hdwig.open(path)

    def fail_second(path, **kwargs):
        if path == "missing":
            raise OSError("cannot open")
        return track

    monkeypatch.setattr(hdwig, "open", fail_second)
    with pytest.raises(OSError, match="cannot open"):
        hdwig.merge(out, [path, "missing"])
    assert not track.h5.id.valid
    assert not out.exists()


def test_bigwig_closes_inputs_on_open_failure(tmp_path, bigwig, pyBigWig, monkeypatch):
    from unittest.mock import Mock

    bw = pyBigWig.open(bigwig)
    handle = Mock(wraps=bw)
    monkeypatch.setattr(pyBigWig, "open", Mock(side_effect=[handle, OSError("cannot open")]))
    out = tmp_path / "out.hw"
    with pytest.raises(OSError, match="cannot open"):
        convert.from_bigwig([bigwig, "missing"], out)
    handle.close.assert_called_once_with()
    assert not out.exists()


def test_bedgraph_export_precision_and_resolution(tmp_path):
    path, out = tmp_path / "a.hw", tmp_path / "out.bg"
    x = np.array([0, 1e-7, 1.2345678, np.nan, np.inf], "float32")
    hdwig.write(path, {"c": x}, dtype="float32", resolution=10)
    convert.to_bedgraph(path, out)
    rows = [line.split() for line in out.read_text().splitlines()]
    assert [row[:3] for row in rows] == [["c", "10", "20"], ["c", "20", "30"]]
    np.testing.assert_array_equal(np.array([row[3] for row in rows], "float32"), x[1:3])


def test_bigwig_export_resolution(tmp_path, pyBigWig):
    path, out = tmp_path / "a.hw", tmp_path / "out.bw"
    hdwig.write(path, {"c": np.array([0, 2, 2, 0], "float32")}, resolution=10)
    convert.to_bigwig(path, out)
    with pyBigWig.open(str(out)) as bw:
        assert bw.chroms() == {"c": 40}
        assert bw.intervals("c") == ((10, 30, 2.0),)


def test_cli_merge_metadata_and_no_compression(tmp_path):
    path, out = tmp_path / "a.hw", tmp_path / "out.hw"
    hdwig.write(path, {"c": np.ones(4)}, resolution=10, units="reads")
    cli.main(["merge", str(out), str(path), "--compression", "none"])
    with hdwig.open(out) as track:
        assert (track.resolution, track.units) == (10, "reads")
        assert track.h5["c"].id.get_create_plist().get_nfilters() == 0
