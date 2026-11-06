"""Isolated ABC invocations. Mapping never overwrites a search state."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re
import subprocess
import shutil
import tempfile
from time import perf_counter


class ABCError(RuntimeError):
    """A tool command failed or did not produce its promised artifact."""


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote(path):
    text = str(Path(path).resolve())
    if any(c in text for c in '\n\r;"\\'):
        raise ValueError("Unsupported character in ABC path")
    return '"' + text + '"'


@dataclass(frozen=True)
class Action:
    name: str
    command: str


def actions(seed=0):
    """Reroll once per decision; a tree edge always retains its exact command."""
    rng = random.Random(seed)
    result = [Action(c, c) for c in (
        "rewrite", "rewrite -z", "balance", "refactor", "refactor -z", "resub -z")]
    k = rng.choice((6, 8, 10, 12))
    resub = f"resub -K {k}" + (" -N 2" if rng.random() < .5 else "")
    result.append(Action(resub, resub))
    dch = rng.choice(("&dch", "&dch -f"))
    macro = f"&get; {dch}; &if -a -K {rng.choice((3, 4, 5))}; &mfs -e -W 20 -L 20; &fx; &st; &put; balance"
    result.append(Action("LUT macro", macro))
    return tuple(result)


class ABC:
    def __init__(self, executable, library, timeout=60):
        self.executable = str(Path(executable).resolve())
        self.library = Path(library).resolve()
        self.library_sha256 = fingerprint(self.library)
        self.timeout = timeout
        self.liberty = self.library.suffix.lower() == '.lib'
        self.mapping_profile = ({'schema': 'abc-liberty-map-stime-v1',
            'library_format': 'liberty', 'mapping': 'map', 'timing': 'topo; stime',
            'input_driver': 'BUF_X1', 'output_load_ff': 10., 'wire_load': False,
            'area_unit': 'um^2', 'delay_unit': 'ps'} if self.liberty else
            {'schema': 'abc-genlib-map-v1', 'library_format': 'genlib',
             'mapping': 'map', 'timing': 'print_stats', 'area_unit': 'library', 'delay_unit': 'library'})
        self.mapping_calls = 0
        self.transform_calls = 0
        self.eda_commands = 0
        self.verification_calls = 0
        self.verification_seconds = 0.
        self.process_calls = 0
        self.process_seconds = 0.

    def run(self, commands, workdir, *, timeout=None):
        # -s suppresses startup scripts/aliases from the caller's home directory.
        started = perf_counter()
        self.process_calls += 1
        try:
            result = subprocess.run([self.executable, "-s", "-c", "; ".join(commands)],
                                    cwd=workdir, capture_output=True, text=True,
                                    timeout=self.timeout if timeout is None else timeout)
        finally:
            self.process_seconds += perf_counter() - started
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
        # ABC sometimes exits zero after rejecting a command.
        if result.returncode or re.search(
                r"(?im)(^Error:|unknown command|cannot open|cannot read|command .* failed|^usage:)", output):
            raise ABCError(f"ABC failed ({result.returncode}):\n{output[-4000:]}")
        return output

    def prepare(self, source, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        reader = "read_truth -xf" if Path(source).suffix == ".truth" else "read"
        self.run([f"{reader} {quote(source)}", "strash",
                  f"write_aiger {quote(destination)}"], destination.parent)
        self._require(destination)
        return destination

    def apply(self, source, action, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        self.transform_calls += 1
        self.eda_commands += len([part for part in action.command.split(";") if part.strip()])
        self.run([f"read {quote(source)}", action.command,
                  f"write_aiger {quote(destination)}"], destination.parent)
        self._require(destination)
        return destination

    def library_commands(self, path=None):
        path = self.library if path is None else path
        return [f"{'read_lib' if self.liberty else 'read_library'} {quote(path)}"]

    def timing_commands(self, directory):
        if not self.liberty:
            return ['print_stats']
        constraints = Path(directory) / 'timing.constr'
        constraints.write_text('set_driving_cell BUF_X1\nset_load 10\n')
        return [f'read_constr {quote(constraints)}', 'topo', 'stime']

    def validate_profile(self, payload):
        recorded = payload.get('mapping_profile')
        if recorded != self.mapping_profile and (recorded is not None or self.liberty):
            raise ValueError('Checkpoint mapping/timing profile mismatch; collect labels and retrain')

    def map(self, state):
        self.mapping_calls += 1
        mapped = Path(state).with_suffix(f".mapped-{self.mapping_calls}.blif")
        if mapped.exists():
            raise FileExistsError(mapped)
        # ABC's generated supergate header tokenizes the library name at spaces.
        # A fixed relative name also isolates auxiliary .super files from user data.
        with tempfile.TemporaryDirectory(prefix="aig-map-") as directory:
            name = "cells.lib" if self.liberty else "cells.genlib"
            shutil.copyfile(self.library, Path(directory) / name)
            reader = "read_lib" if self.liberty else "read_library"
            output = self.run([f"{reader} {name}", f"read {quote(state)}", "map",
                               *self.timing_commands(directory), f"write_blif {quote(mapped)}"], directory)
        area, delay = self._mapped_metrics(output)
        self._require(mapped)
        # ABC derives the BLIF model name from the AIG's absolute path. Spaces in
        # that path otherwise produce an unreadable .model declaration.
        mapped.write_text(re.sub(r"(?m)^\.model[^\n]*", ".model mapped_design", mapped.read_text(), count=1))
        return {"area": area, "delay": delay, "netlist": mapped}

    @staticmethod
    def _mapped_metrics(output):
        matches = re.findall(r"area\s*=\s*([\d.eE+-]+).*?delay\s*=\s*([\d.eE+-]+)", output, re.I)
        if not matches:
            raise RuntimeError(f"ABC did not report mapped area and delay:\n{output}")
        area, delay = map(float, matches[-1])
        if not math.isfinite(area) or not math.isfinite(delay) or area <= 0 or delay <= 0:
            raise ValueError("Mapped metrics must be positive; constant-only circuits are unsupported")
        return area, delay

    def measure(self, netlist):
        """Read mapped-netlist cost without resynthesizing or writing artifacts."""
        netlist = Path(netlist).resolve()
        with tempfile.TemporaryDirectory(prefix='aig-measure-') as directory:
            output = self.run([*self.library_commands(), f"read {quote(netlist)}",
                               *self.timing_commands(directory)], directory)
        area, delay = self._mapped_metrics(output)
        return {'area': area, 'delay': delay}

    def verify(self, reference, candidate, *, timeout=None):
        self.verification_calls += 1
        started = perf_counter()
        try:
            output = self.run([*self.library_commands(),
                               f"cec {quote(reference)} {quote(candidate)}"], Path(candidate).parent,
                              timeout=timeout)
        finally:
            self.verification_seconds += perf_counter() - started
        if "Networks are equivalent" not in output or "NOT EQUIVALENT" in output:
            raise RuntimeError(f"Equivalence was not proven:\n{output}")
        return True

    @staticmethod
    def _require(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ABCError(f"ABC did not create {path.name}")
