"""Integration tests: the contract, rather than the current output strings.

This library exists to take an ``ArgumentParser``, describe it, and rebuild a
command line the SAME parser accepts. The existing unit tests assert on literal
token lists, which pins the implementation but never checks that property -- and
that is how two import-time codegen bugs and an unparseable command line all
survived 100% line AND branch coverage.

Two properties are checked here:

  round-trip    build_cmdline_args(values) -> parser.parse_args() -> values
  codegen       the emitted module imports, and describes the same parser
"""

from __future__ import annotations

import argparse
from enum import Enum

import pytest

from argparse_schema import (
    build_cmdline_args,
    extract_default_args,
    generate_cli_metadata_code,
    generate_dataclass,
    generate_defaults_yaml,
    get_action_specs,
    get_arg_metadata,
)


class Mode(Enum):
    FAST = "fast"
    SLOW = "slow"


def make_parser() -> argparse.ArgumentParser:
    """A parser using every action type this library claims to support."""
    p = argparse.ArgumentParser(prog="train", allow_abbrev=False)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--name", type=str, default="run")
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--tags", type=str, nargs="*", default=[])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.add_argument("--verbose", "-v", action="count", default=0)
    p.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
    return p


@pytest.fixture
def parser():
    return make_parser()


@pytest.fixture
def schema(parser):
    return get_arg_metadata(parser), get_action_specs(parser)


def roundtrip(parser, schema, values: dict, **kwargs) -> dict:
    """Emit a command line for ``values`` and parse it back with ``parser``."""
    metadata, specs = schema
    argv = build_cmdline_args(values, metadata, specs, **kwargs)
    # parse_args exits the process on a bad command line, which as a test failure
    # would say only "SystemExit". Name the argv that caused it instead.
    try:
        parsed = parser.parse_args(argv)
    except SystemExit:  # pragma: no cover - only on a regression
        pytest.fail(f"parser rejected its own generated command line: {argv}")
    return vars(parsed)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param({"lr": 3e-4}, id="float"),
            pytest.param({"steps": 5000}, id="int"),
            pytest.param({"name": "my-run"}, id="str"),
            pytest.param({"name": "with spaces"}, id="str-with-spaces"),
            pytest.param({"name": ""}, id="empty-str"),
            pytest.param({"layers": [4, 5, 6]}, id="nargs-plus"),
            pytest.param({"tags": ["a", "b"]}, id="nargs-star"),
            pytest.param({"fp16": True}, id="store-true"),
            pytest.param({"cache": False}, id="store-false"),
            pytest.param({"verbose": 3}, id="count"),
            pytest.param({"opt": "sgd"}, id="choices"),
            pytest.param({"lr": 2e-4, "steps": 7, "fp16": True}, id="several-at-once"),
        ],
    )
    def test_values_survive_the_round_trip(self, parser, schema, values):
        parsed = roundtrip(parser, schema, values)
        for key, value in values.items():
            assert parsed[key] == value, f"{key}: {parsed[key]!r} != {value!r}"

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param({"name": "-x"}, id="leading-dash"),
            pytest.param({"name": "--looks-like-a-flag"}, id="double-dash"),
            pytest.param({"lr": -1.5}, id="negative-float"),
            pytest.param({"steps": -3}, id="negative-int"),
        ],
    )
    def test_dash_prefixed_values_survive(self, parser, schema, values):
        """A value starting with "-" is read as an option in the two-token form.

        argparse errors with "expected one argument" and never assigns it; the
        ``--opt=value`` form is unambiguous.
        """
        parsed = roundtrip(parser, schema, values)
        for key, value in values.items():
            assert parsed[key] == value

    def test_defaults_emit_nothing_by_default(self, parser, schema):
        """skip_defaults is the point: an unchanged value adds no tokens."""
        metadata, specs = schema
        defaults = extract_default_args(parser)
        assert build_cmdline_args(defaults, metadata, specs, skip_defaults=True) == []

    def test_defaults_still_round_trip_when_emitted(self, parser, schema):
        metadata, specs = schema
        defaults = extract_default_args(parser)
        argv = build_cmdline_args(defaults, metadata, specs, skip_defaults=False)
        parsed = vars(parser.parse_args(argv))
        for key, value in defaults.items():
            assert parsed[key] == value, f"{key}: {parsed[key]!r} != {value!r}"

    def test_list_sep_joins_into_one_token(self, parser, schema):
        """``list_sep`` targets tools that want ``--opt a,b``, not argparse.

        argparse's own nargs wants separate tokens, so this is deliberately NOT
        a round-trip through ``parser``: it is checked as the shape it promises.
        """
        metadata, specs = schema
        argv = build_cmdline_args({"layers": [4, 5]}, metadata, specs, list_sep=",")
        assert argv == ["--layers", "4,5"]

    def test_nargs_element_types_survive_coercion(self, parser, schema):
        """Elements keep the parser's type instead of collapsing to str.

        Not cosmetic: ``skip_defaults`` compares against the parser's default, so
        stringified elements made every nargs default compare unequal and be
        emitted on every command line.
        """
        metadata, specs = schema
        assert build_cmdline_args({"layers": [1, 2]}, metadata, specs) == []
        assert build_cmdline_args({"layers": [1, 3]}, metadata, specs) == [
            "--layers",
            "1",
            "3",
        ]


class TestGeneratedCliMetadata:
    """The emitted module must import, and describe the parser it came from."""

    def test_module_executes_and_matches_the_parser(self, parser, schema, tmp_path):
        metadata, specs = schema
        module = tmp_path / "generated.py"
        module.write_text(generate_cli_metadata_code(metadata, specs))

        namespace: dict = {}
        exec(compile(module.read_text(), str(module), "exec"), namespace)

        assert set(namespace["ARG_METADATA"]) == set(metadata)
        assert set(namespace["ACTION_SPECS"]) == set(specs)

    def test_generated_metadata_rebuilds_the_same_command_line(self, parser, schema, tmp_path):
        """The whole point of emitting it: it must work in place of the parser."""
        metadata, specs = schema
        module = tmp_path / "generated.py"
        module.write_text(generate_cli_metadata_code(metadata, specs))
        namespace: dict = {}
        exec(compile(module.read_text(), str(module), "exec"), namespace)

        values = {"lr": 3e-4, "steps": 42, "fp16": True, "layers": [7, 8]}
        from_parser = build_cmdline_args(values, metadata, specs)
        from_generated = build_cmdline_args(
            values, namespace["ARG_METADATA"], namespace["ACTION_SPECS"]
        )
        assert from_generated == from_parser
        assert vars(parser.parse_args(from_generated))["steps"] == 42

    def test_enum_typed_argument_does_not_break_the_module(self, tmp_path):
        """An Enum reprs as ``<Mode.FAST: 'fast'>`` and names an unimportable type.

        Both used to reach the generated file verbatim, so it failed at import --
        far from the codegen call, and only for parsers that happen to use one.
        """
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument("--mode", type=Mode, default=Mode.FAST)
        p.add_argument("--steps", type=int, default=1)
        code = generate_cli_metadata_code(get_arg_metadata(p), get_action_specs(p))

        namespace: dict = {}
        exec(compile(code, "<gen>", "exec"), namespace)  # must not raise

        # The enum becomes the token a command line actually carries.
        assert namespace["ARG_METADATA"]["mode"].default == "fast"
        # Untouched neighbours keep their real types.
        assert namespace["ARG_METADATA"]["steps"].arg_type is int

    def test_unrepresentable_default_fails_loudly_at_codegen(self):
        """Better than emitting a file that cannot be imported."""

        class Weird:
            pass

        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument("--w", default=Weird())
        with pytest.raises(ValueError, match="cannot emit a literal for w.default"):
            generate_cli_metadata_code(get_arg_metadata(p), get_action_specs(p))


class TestGeneratedDataclassAndYaml:
    """The other two emitters produce artefacts nothing was checking either."""

    def test_generated_dataclass_is_valid_python(self, parser):
        code = generate_dataclass(get_arg_metadata(parser), extract_default_args(parser))
        compile(code, "<gen>", "exec")  # must not raise

    def test_generated_dataclass_covers_every_argument(self, parser):
        defaults = extract_default_args(parser)
        code = generate_dataclass(get_arg_metadata(parser), defaults)
        for dest in defaults:
            assert f"{dest}:" in code, f"{dest} missing from the generated dataclass"

    def test_generated_dataclass_survives_an_enum(self):
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument("--mode", type=Mode, default=Mode.FAST)
        code = generate_dataclass(get_arg_metadata(p), extract_default_args(p))
        compile(code, "<gen>", "exec")
        assert "mode:" in code

    def test_generated_yaml_parses_and_covers_every_argument(self, parser):
        yaml = pytest.importorskip("yaml")
        defaults = extract_default_args(parser)
        loaded = yaml.safe_load(generate_defaults_yaml(get_arg_metadata(parser), defaults))
        assert set(loaded) == set(defaults)
