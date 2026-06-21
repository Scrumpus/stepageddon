"""Round-trip test for the .sm exporter.

Parse a known simfile → export it → re-parse the export → assert the notes
survive. Proves the seconds→beats inverse in :mod:`src.charts.sm_export` mirrors
the importer in :mod:`src.songs.utils`.

Runs under pytest, or directly: ``python tests/test_sm_export.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charts.sm_export import build_simfile, build_song_zip  # noqa: E402
from src.songs.utils import parse_sm  # noqa: E402

# A simfile with a BPM change, jumps, and a hold — the tricky cases.
SAMPLE_SM = """#TITLE:Round Trip;
#ARTIST:Tester;
#MUSIC:song.ogg;
#OFFSET:-0.012;
#BPMS:0.000=120.000,4.000=150.000;
#NOTES:
     dance-single:
     :
     Hard:
     8:
     :
1001
0000
0100
0010
,
2000
0000
3000
0101
;
"""


def _normalize(chart):
    return [
        (
            round(s.time, 2),
            tuple(sorted(a.value for a in s.arrows)),
            s.step_type.value,
            round(s.hold_duration, 2) if s.hold_duration else None,
        )
        for s in chart.steps
    ]


def test_sm_export_roundtrip():
    src = Path(tempfile.mktemp(suffix=".sm"))
    src.write_text(SAMPLE_SM)
    parsed = parse_sm(src)
    assert parsed.charts, "fixture failed to parse"
    chart = parsed.charts[0]

    steps = [
        {
            "time": s.time,
            "arrows": [a.value for a in s.arrows],
            "type": s.step_type.value,
            "hold_duration": s.hold_duration,
        }
        for s in chart.steps
    ]
    meta = {
        "title": parsed.title,
        "artist": parsed.artist,
        "music": "song.ogg",
        "offset": parsed.offset,
        "bpms": [list(b) for b in parsed.bpms],
    }
    chart_dicts = [
        {
            "difficulty_name": chart.difficulty_name,
            "difficulty_level": chart.difficulty_level,
            "steps": steps,
            "radar_values": chart.radar_values,
        }
    ]

    sm_text = build_simfile(meta, chart_dicts)
    out = Path(tempfile.mktemp(suffix=".sm"))
    out.write_text(sm_text)
    reparsed = parse_sm(out)

    assert reparsed.charts, "export failed to re-parse"
    assert _normalize(chart) == _normalize(reparsed.charts[0])
    assert reparsed.title == parsed.title
    assert abs(reparsed.offset - parsed.offset) < 1e-6


def test_build_song_zip_layout():
    zip_bytes = build_song_zip("#TITLE:X;", "My Song", {"My Song.ogg": b"AUDIO"})
    import io
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(zip_bytes)).namelist()
    assert "My Song/My Song.sm" in names
    assert "My Song/My Song.ogg" in names


if __name__ == "__main__":
    test_sm_export_roundtrip()
    test_build_song_zip_layout()
    print("OK: all sm_export round-trip tests passed")
