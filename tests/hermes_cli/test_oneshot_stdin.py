import io

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.oneshot import read_oneshot_stdin


def test_oneshot_stdin_parser_is_mutually_exclusive_with_literal_oneshot():
    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["--oneshot-stdin", "--toolsets", "context_engine"])
    assert args.oneshot_stdin is True
    assert args.oneshot is None

    with pytest.raises(SystemExit):
        parser.parse_args(["--oneshot-stdin", "-z", "literal"])


def test_read_oneshot_stdin_is_bounded_and_never_reflects_payload():
    assert read_oneshot_stdin(io.StringIO("private\ninput"), limit_bytes=32) == "private\ninput"
    with pytest.raises(ValueError, match="input limit") as exc:
        read_oneshot_stdin(io.StringIO("do-not-reflect"), limit_bytes=4)
    assert "do-not-reflect" not in str(exc.value)

    with pytest.raises(ValueError, match="empty"):
        read_oneshot_stdin(io.StringIO(""))
