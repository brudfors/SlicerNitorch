"""Install nitorch into 3D Slicer's Python, preferring the fast compiled backend.

Run from Slicer's Python console (``View > Python Console``)::

    import sys; sys.path.insert(0, "/path/to/SlicerNitorch")
    import install_nitorch
    install_nitorch.run()                  # default: install torch matched to your CUDA
                                           # toolkit (via SlicerPyTorch) + compiled nitorch
    install_nitorch.run(align_torch=False) # never change/choose torch CUDA; compiled only
                                           # if torch already matches a toolkit, else pure
    install_nitorch.run(force="pure")      # explicit pure (TorchScript) backend

Torch itself is always managed by the SlicerPyTorch extension (PyTorchUtils): this
script detects the newest CUDA toolkit installed on the system and asks the extension
to install/realign torch to match, so the compiled backend can be built. Plain pip is
used only for ``torch_interpol`` and ``nitorch``. Every failure prints an actionable
message and falls back to the pure backend so the install always ends up working.
"""

import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys

NITORCH_REF = "git+https://github.com/balbasty/nitorch.git@master"
_PREFIX = "[install_nitorch]"


def _log(msg):
    print(f"{_PREFIX} {msg}")


def _python_slicer():
    """Locate the PythonSlicer executable for the running Slicer."""
    candidates = []
    try:
        import slicer
        home = getattr(slicer.app, "slicerHome", None)
        if home:
            candidates.append(os.path.join(home, "bin", "PythonSlicer"))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.dirname(sys.executable), "PythonSlicer"))
    which = shutil.which("PythonSlicer")
    if which:
        candidates.append(which)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise RuntimeError(
        "Could not locate PythonSlicer. Run this from Slicer's Python console.")


def _run_pip(args, env=None):
    """Run `PythonSlicer -m pip install <args>`. Returns (rc, combined_output)."""
    cmd = [_python_slicer(), "-m", "pip", "install", *args]
    _log("pip " + " ".join(["install", *args]))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _cuda_toolkits():
    """Return [((major, minor), cuda_home), ...] for /usr/local/cuda-*/bin/nvcc, ascending."""
    found = []
    for nvcc in glob.glob("/usr/local/cuda-*/bin/nvcc"):
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
            m = re.search(r"release (\d+)\.(\d+)", r.stdout)
            if m:
                ver = (int(m.group(1)), int(m.group(2)))
                home = os.path.dirname(os.path.dirname(nvcc))
                found.append((ver, home))
        except Exception:
            pass
    found.sort()
    return found


def _torch_present():
    return importlib.util.find_spec("torch") is not None


def _torch_cuda():
    """torch's CUDA version as (major, minor), or None if torch is CPU-only."""
    import torch
    cu = torch.version.cuda
    if not cu:
        return None
    parts = cu.split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _backend_code(ver):
    return f"cu{ver[0]}{ver[1]}"


def _pick_compiler():
    """Pick an existing (gcc, g++) pair, newest first. Slicer's Python records a
    build compiler (e.g. a Red Hat gcc-toolset) that usually isn't present, so we must
    point CC/CXX at a compiler that actually exists. Returns (cc, cxx) or (None, None).
    """
    for suffix in ("-14", "-13", "-12", "-11", "-10", ""):
        cc = shutil.which("gcc" + suffix)
        cxx = shutil.which("g++" + suffix)
        if cc and cxx:
            return cc, cxx
    return None, None


def _ensure_python_headers():
    """Ensure CPython development headers are available for compiling extensions.

    Slicer's bundled Python ships ``pyconfig.h`` but not the rest of the CPython
    headers (no ``Python.h``), so C extensions can't compile out of the box. If
    ``Python.h`` is missing from the Python include dir, download the matching CPython
    source headers from python.org and copy them in (without clobbering Slicer's
    ``pyconfig.h``). Returns True if headers are present (already or after fetch).
    """
    import sysconfig
    import platform
    inc = sysconfig.get_path("include")
    if os.path.exists(os.path.join(inc, "Python.h")):
        return True
    ver = platform.python_version()  # e.g. "3.12.10"
    url = f"https://www.python.org/ftp/python/{ver}/Python-{ver}.tgz"
    _log(f"Python.h not found in {inc}; fetching CPython {ver} headers from python.org "
         "(Slicer ships only pyconfig.h)...")
    try:
        import urllib.request
        import tarfile
        import tempfile
        tmp = tempfile.mkdtemp()
        tgz = os.path.join(tmp, "py.tgz")
        urllib.request.urlretrieve(url, tgz)
        prefix = f"Python-{ver}/Include/"
        with tarfile.open(tgz) as tf:
            members = [m for m in tf.getmembers() if m.name.startswith(prefix)]
            for m in members:
                tf.extract(m, tmp, filter="data")
        src_inc = os.path.join(tmp, f"Python-{ver}", "Include")
        for root, _dirs, files in os.walk(src_inc):
            rel = os.path.relpath(root, src_inc)
            dst_root = inc if rel == "." else os.path.join(inc, rel)
            os.makedirs(dst_root, exist_ok=True)
            for f in files:
                dst = os.path.join(dst_root, f)
                if not os.path.exists(dst):  # keep Slicer's own headers (pyconfig.h)
                    shutil.copy2(os.path.join(root, f), dst)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        _log(f"Failed to fetch/install Python headers: {e}")
        return False
    if os.path.exists(os.path.join(inc, "Python.h")):
        _log(f"Installed CPython {ver} headers into {inc}.")
        return True
    _log("Python.h still missing after fetch; cannot compile.")
    return False


def _verify():
    """Return the active nitorch backend ('C' = compiled, else 'TS'), from a fresh process."""
    env = os.environ.copy()
    env["NI_CHECK_BACKEND"] = "0"  # suppress the TS warning so output is clean
    code = "import nitorch; print('BACKEND=' + str(nitorch.compiled_backend))"
    r = subprocess.run([_python_slicer(), "-c", code], capture_output=True, text=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"BACKEND=(\S+)", out)
    return m.group(1) if m else "?"


def _build(compiled, cuda_home=None):
    """Install nitorch. compiled=True builds the C backend against the installed torch."""
    # --force-reinstall so a TS->compiled (or vice-versa) switch actually takes effect
    # (same version string, different wheel contents); --no-deps so we don't disturb the
    # torch we carefully installed (deps are installed separately in run()).
    args = ["--no-build-isolation", "--force-reinstall", "--no-deps", NITORCH_REF]
    env = None
    if compiled:
        if not _ensure_python_headers():
            _log("Could not obtain Python headers; installing the TS backend instead.")
            return _build(compiled=False)
        env = os.environ.copy()
        env["NI_COMPILED_BACKEND"] = "C"
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + env.get("PATH", "")
        cc, cxx = _pick_compiler()
        if cxx:
            env["CC"], env["CXX"] = cc, cxx
            _log(f"Using compiler CC={cc}, CXX={cxx}.")
        else:
            _log("WARNING: no gcc/g++ found on PATH; the build will use Python's recorded "
                 "compiler, which may be missing. Install g++ (build-essential).")
        _log(f"Building the COMPILED (C) backend (CUDA_HOME={cuda_home}). This compiles "
             "many CPU + CUDA kernels and can take a LONG time — 20-30 minutes is not "
             "unusual; please don't interrupt it.")
    else:
        _log("Installing the non-compiled (TS / TorchScript) backend...")
    rc, out = _run_pip(args, env=env)
    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        _log("pip build output (tail):\n" + tail)
        if compiled:
            _log("Compiled build FAILED (see above). Common causes: missing/old g++ "
                 "(install build-essential or a newer g++), or a CUDA toolkit/torch "
                 "version mismatch. Falling back to the non-compiled (TS) backend.")
            return _build(compiled=False)
        raise RuntimeError("nitorch install failed; see pip output above.")
    return compiled


def _install_torch_via_extension(backend, torch_version, reinstall):
    """Install/realign torch using the SlicerPyTorch extension. Returns True on success."""
    try:
        import PyTorchUtils
    except ImportError:
        _log("The SlicerPyTorch extension isn't installed (no PyTorchUtils). Install it "
             "via Extensions Manager > PyTorch, then re-run.")
        return False
    logic = PyTorchUtils.PyTorchUtilsLogic()
    if reinstall:
        logic.uninstallTorch()
    logic.installTorch(askConfirmation=False, forceComputationBackend=backend,
                       torchVersionRequirement=torch_version)
    return True


def run(align_torch=True, force=None, torch_version=None):
    """Install torch_interpol + nitorch into Slicer's Python (compiled backend by default).

    Parameters
    ----------
    align_torch : bool
        If True (default), install/realign torch (via the SlicerPyTorch extension) to
        match the newest CUDA toolkit found on the system, so the compiled backend can
        be built. If False, torch is never installed/changed by this script for the
        sake of matching (compiled is built only if torch already matches a toolkit).
    force : {None, 'pure', 'compiled'}
        Force a backend instead of auto-selecting.
    torch_version : str or None
        Optional version requirement for the SlicerPyTorch installer, e.g. '==2.9.1'.
    """
    if force not in (None, "pure", "compiled"):
        _log(f"Unknown force={force!r}; expected None, 'pure', or 'compiled'.")
        return

    _log("Starting nitorch install for Slicer.")

    # The compiled backend is only supported on Linux by this installer.
    # TODO: native compiled-backend builds for Windows/macOS (different toolchains;
    # macOS has no CUDA). For now those platforms use the TS backend.
    if sys.platform != "linux":
        if force == "compiled":
            _log(f"force='compiled' is not supported on '{sys.platform}' "
                 "(compiled backend is Linux-only here).")
        _log(f"Platform '{sys.platform}': installing the non-compiled (TS) backend "
             "(the compiled backend is currently Linux-only).")
        force = "pure"
    toolkits = _cuda_toolkits()
    gpp = shutil.which("g++")
    compile_capable = bool(toolkits) and bool(gpp) and force != "pure"
    target_backend = _backend_code(toolkits[-1][0]) if toolkits else None
    if toolkits:
        v = toolkits[-1][0]
        _log(f"Newest CUDA toolkit: {v[0]}.{v[1]} ({toolkits[-1][1]}); g++: {gpp or 'NOT FOUND'}.")
    else:
        _log(f"No CUDA toolkit found under /usr/local/cuda-*; g++: {gpp or 'NOT FOUND'}.")

    # 1. Ensure PyTorch is present, matched to the toolkit when we intend to compile.
    if not _torch_present():
        if compile_capable and align_torch:
            v = toolkits[-1][0]
            _log(f"PyTorch not found; installing torch ({target_backend}) to match CUDA "
                 f"toolkit {v[0]}.{v[1]} via the SlicerPyTorch extension (~2 GB)...")
            ok = _install_torch_via_extension(target_backend, torch_version, reinstall=False)
        else:
            _log("PyTorch not found; installing the driver-default torch via the "
                 "SlicerPyTorch extension (~2 GB)...")
            ok = _install_torch_via_extension(None, torch_version, reinstall=False)
        if not ok:
            return

    import torch
    tcuda = _torch_cuda()
    _log(f"Found torch {torch.__version__} (CUDA {torch.version.cuda}, "
         f"available={torch.cuda.is_available()}).")

    # 2. nitorch runtime deps (installed here since the build below uses --no-deps) and
    #    build deps (needed for the --no-build-isolation nitorch build).
    _run_pip(["torch_interpol>=0.3.0", "numpy", "scipy"])
    _run_pip(["versioneer", "wheel", "setuptools"])

    # 3. Backend selection + build.
    def finish():
        backend = _verify()
        if backend == "C":
            _log("Done — nitorch COMPILED (C) backend is active.")
        else:
            _log(f"Done — nitorch non-compiled backend ({backend}); "
                 "some algorithms may be slower.")
        _log("Restart Slicer for the module to pick up the new nitorch.")

    if force == "pure":
        _build(compiled=False); finish(); return
    if not toolkits:
        _log("Compiled backend needs a CUDA toolkit (nvcc); none found. Installing the "
             "TS backend now (install a CUDA toolkit to enable the faster compiled one).")
        _build(compiled=False); finish(); return
    if not gpp:
        _log("Compiled backend needs a C++ compiler; g++ not found (install "
             "build-essential). Installing the TS backend now.")
        _build(compiled=False); finish(); return
    if tcuda is None:
        _log("torch is CPU-only; the compiled CUDA backend does not apply. Installing "
             "the TS backend.")
        _build(compiled=False); finish(); return

    newest_ver, newest_home = toolkits[-1]
    if newest_ver == tcuda or newest_ver[0] == tcuda[0]:
        if newest_ver != tcuda:
            _log(f"Note: toolkit {newest_ver} vs torch CUDA {tcuda} (minor differs); "
                 "proceeding since major versions match.")
        _build(compiled=True, cuda_home=newest_home); finish(); return

    if align_torch:
        _log(f"torch is CUDA {tcuda[0]}.{tcuda[1]} but newest toolkit is "
             f"{newest_ver[0]}.{newest_ver[1]}; reinstalling torch as "
             f"{_backend_code(newest_ver)} to match (~2 GB)...")
        if not _install_torch_via_extension(_backend_code(newest_ver), torch_version,
                                            reinstall=True):
            _log("Could not realign torch; installing the TS backend instead.")
            _build(compiled=False); finish(); return
        _build(compiled=True, cuda_home=newest_home); finish(); return

    _log(f"torch is CUDA {tcuda[0]}.{tcuda[1]} but no matching CUDA toolkit is installed "
         f"(have {newest_ver[0]}.{newest_ver[1]}) and align_torch=False. Install a "
         "matching toolkit or allow aligning. Installing the TS backend now.")
    _build(compiled=False); finish()
