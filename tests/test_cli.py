"""Basic CLI tests — validates the Being CLI loads and shows help."""

from click.testing import CliRunner
from being.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Being" in result.output or "being" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_prepare_help():
    runner = CliRunner()
    result = runner.invoke(main, ["prepare", "--help"])
    assert result.exit_code == 0
    assert "video_path" in result.output.lower() or "VIDEO_PATH" in result.output


def test_train_help():
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0


def test_serve_help():
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0


def test_generate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["generate", "--help"])
    assert result.exit_code == 0
