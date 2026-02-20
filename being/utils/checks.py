"""Dependency checker — validates that all required tools and libraries are available."""

import sys
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_checks():
    """Run all dependency checks and print a status report."""
    console.print("[bold]Being — Dependency Check[/]\n")

    table = Table(title="Dependencies")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    checks = [
        check_python,
        check_cuda,
        check_pytorch,
        check_instag,
        check_gaussian_rasterizer,
        check_ffmpeg,
        check_openface,
        check_bfm,
        check_easyportrait,
    ]

    all_ok = True
    for check_fn in checks:
        name, ok, detail = check_fn()
        status = "[green]OK" if ok else "[red]MISSING"
        if not ok:
            all_ok = False
        table.add_row(name, status, detail)

    console.print(table)

    if all_ok:
        console.print("\n[bold green]All dependencies satisfied![/]")
    else:
        console.print("\n[bold yellow]Some dependencies are missing. Run `bash scripts/setup.sh` to install.[/]")


def check_python() -> tuple[str, bool, str]:
    v = sys.version_info
    ok = v.major == 3 and 9 <= v.minor <= 10
    return "Python 3.9-3.10", ok, f"{v.major}.{v.minor}.{v.micro}"


def check_cuda() -> tuple[str, bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return "CUDA", True, f"CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}"
        return "CUDA", False, "CUDA not available"
    except ImportError:
        return "CUDA", False, "PyTorch not installed"


def check_pytorch() -> tuple[str, bool, str]:
    try:
        import torch
        return "PyTorch", True, f"{torch.__version__}"
    except ImportError:
        return "PyTorch", False, "Not installed"


def check_instag() -> tuple[str, bool, str]:
    instag_dir = PROJECT_ROOT / "extern" / "InsTaG"
    if (instag_dir / "synthesize_fuse.py").exists():
        return "InsTaG", True, str(instag_dir)
    return "InsTaG", False, "Run: git submodule update --init --recursive"


def check_gaussian_rasterizer() -> tuple[str, bool, str]:
    try:
        import diff_gaussian_rasterization
        return "diff-gaussian-rasterization", True, "Installed"
    except ImportError:
        return "diff-gaussian-rasterization", False, "CUDA extension not compiled"


def check_ffmpeg() -> tuple[str, bool, str]:
    if shutil.which("ffmpeg"):
        return "ffmpeg", True, shutil.which("ffmpeg")
    return "ffmpeg", False, "Not in PATH"


def check_openface() -> tuple[str, bool, str]:
    for path in [
        shutil.which("FeatureExtraction"),
        "/usr/local/bin/FeatureExtraction",
        "/opt/OpenFace/build/bin/FeatureExtraction",
    ]:
        if path and Path(path).exists():
            return "OpenFace", True, str(path)
    return "OpenFace", False, "Optional — install from github.com/TadasBaltrusaitis/OpenFace"


def check_bfm() -> tuple[str, bool, str]:
    bfm_dir = PROJECT_ROOT / "extern" / "InsTaG" / "data_utils" / "face_tracking" / "3DMM"
    if (bfm_dir / "01_MorphableModel.mat").exists() or (bfm_dir / "BFM09_model_info.mat").exists():
        return "Basel Face Model", True, str(bfm_dir)
    return "Basel Face Model", False, "Register at faces.dmi.unibas.ch/bfm"


def check_easyportrait() -> tuple[str, bool, str]:
    model_path = PROJECT_ROOT / "extern" / "InsTaG" / "data_utils" / "easyportrait" / "fpn-fp-512.pth"
    if model_path.exists():
        return "EasyPortrait", True, str(model_path)
    return "EasyPortrait", False, "Run: bash scripts/setup.sh"
