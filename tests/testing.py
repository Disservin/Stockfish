import subprocess
from typing import List, Callable, Any
import os
import collections
import time
import sys
import traceback
import fnmatch
from functools import wraps
from contextlib import redirect_stdout
import io
import tarfile
import urllib.request
import pathlib
import concurrent.futures
import tempfile

CYAN_COLOR = "\033[36m"
GRAY_COLOR = "\033[2m"
RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
RESET_COLOR = "\033[0m"
WHITE_BOLD = "\033[1m"

MAX_TIMEOUT = 60 * 5

PATH = pathlib.Path(__file__).parent.resolve()


class Valgrind:
    @staticmethod
    def get_valgrind_command():
        return [
            "valgrind",
            "--error-exitcode=42",
            "--errors-for-leak-kinds=all",
            "--leak-check=full",
        ]

    @staticmethod
    def get_valgrind_thread_command():
        return ["valgrind", "--error-exitcode=42", "--fair-sched=try"]


class TSAN:
    @staticmethod
    def set_tsan_option():
        with open(os.path.join(PATH, "tsan.supp"), "w") as f:
            f.write(
                """
race:Stockfish::TTEntry::read
race:Stockfish::TTEntry::save
race:Stockfish::TranspositionTable::probe
race:Stockfish::TranspositionTable::hashfull
"""
            )

        os.environ["TSAN_OPTIONS"] = f"suppressions={os.path.join(PATH,'tsan.supp')}"

    @staticmethod
    def unset_tsan_option():
        os.environ.pop("TSAN_OPTIONS", None)
        os.remove(os.path.join(PATH, "tsan.supp"))


class EPD:
    @staticmethod
    def create_bench_epd():
        with open(os.path.join(PATH, "bench_tmp.epd"), "w") as f:
            f.write(
                """
Rn6/1rbq1bk1/2p2n1p/2Bp1p2/3Pp1pP/1N2P1P1/2Q1NPB1/6K1 w - - 2 26
rnbqkb1r/ppp1pp2/5n1p/3p2p1/P2PP3/5P2/1PP3PP/RNBQKBNR w KQkq - 0 3
3qnrk1/4bp1p/1p2p1pP/p2bN3/1P1P1B2/P2BQ3/5PP1/4R1K1 w - - 9 28
r4rk1/1b2ppbp/pq4pn/2pp1PB1/1p2P3/1P1P1NN1/1PP3PP/R2Q1RK1 w - - 0 13
"""
            )

    @staticmethod
    def delete_bench_epd():
        os.remove(f"{os.path.join(PATH,'bench_tmp.epd')}")


class Syzygy:
    @staticmethod
    def get_syzygy_path():
        return os.path.abspath("syzygy")

    @staticmethod
    def download_syzygy():
        if not os.path.isdir(os.path.join(PATH, "syzygy")):
            url = "https://api.github.com/repos/niklasf/python-chess/tarball/9b9aa13f9f36d08aadfabff872882f4ab1494e95"
            tarball_path = "/tmp/python-chess.tar.gz"

            urllib.request.urlretrieve(url, tarball_path)

            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall("/tmp")

            os.rename("/tmp/niklasf-python-chess-9b9aa13", os.path.join(PATH, "syzygy"))


class OrderedClassMembers(type):
    @classmethod
    def __prepare__(self, name, bases):
        return collections.OrderedDict()

    def __new__(self, name, bases, classdict):
        classdict["__ordered__"] = [
            key for key in classdict.keys() if key not in ("__module__", "__qualname__")
        ]
        return type.__new__(self, name, bases, classdict)


class TimeoutException(Exception):
    def __init__(self, message: str, timeout: int):
        self.message = message
        self.timeout = timeout


def timeout_decorator(timeout: float):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutException(
                        f"Function {func.__name__} timed out after {timeout} seconds",
                        timeout,
                    )
            return result

        return wrapper

    return decorator


import threading
import tempfile
import os
import time
import sys
import traceback
from contextlib import redirect_stdout
import io
from typing import List


class MiniTestFramework:
    def __init__(self):
        self.passed_test_suites = 0
        self.failed_test_suites = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.output_lock = threading.Lock()  # Lock for managing output
        self.test_results = {}  # To store the results of each test class
        self.output_lines = []  # To store all printed lines
        self.start_time = None
        self.printed_lines = 0  # To track how many lines have been printed

    def has_failed(self):
        return bool(self.failed_test_suites)

    def run(self, classes: List[type]):
        self.start_time = time.time()

        threads = []
        for test_class in classes:
            thread = threading.Thread(target=self._run_test_suite, args=(test_class,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()  # Wait for all threads to complete

        # Final output once all threads have finished
        self._print_summary()

        return bool(self.failed_test_suites)

    def _run_test_suite(self, test_class):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ret = self.__run(test_class, tmpdirname)

            with self.output_lock:
                if ret:
                    self.failed_test_suites += 1
                else:
                    self.passed_test_suites += 1

    def __run(self, test_class, tmpdirname):
        test_instance = test_class()
        test_name = test_instance.__class__.__name__

        test_methods = [
            method for method in test_instance.__ordered__ if method.startswith("test_")
        ]

        if "beforeAll" in dir(test_instance):
            test_instance.beforeAll(tmpdirname)

        with self.output_lock:
            self.test_results[test_name] = [f"Test Suite: {test_name}"]
            self._redraw_status()

        fails = 0

        for method in test_methods:
            buffer = io.StringIO()

            try:
                t0 = time.time()

                # with redirect_stdout(buffer):
                if hasattr(test_instance, "beforeEach"):
                    test_instance.beforeEach()

                getattr(test_instance, method)()

                if hasattr(test_instance, "afterEach"):
                    test_instance.afterEach()

                duration = time.time() - t0

                with self.output_lock:
                    self.passed_tests += 1
                    self.test_results[test_name].append(
                        self._format_success(method, duration)
                    )
                    self._redraw_status()

            except TimeoutException as e:
                with self.output_lock:
                    fails += 1
                    self.failed_tests += 1
                    self.test_results[test_name].append(
                        self._format_failure(method, e.timeout)
                    )
                    self._redraw_status()
            except AssertionError:
                duration = time.time() - t0
                with self.output_lock:
                    fails += 1
                    self.failed_tests += 1
                    traceback_output = "".join(traceback.format_tb(sys.exc_info()[2]))
                    self.test_results[test_name].append(
                        self._format_failure(method, duration, traceback_output)
                    )
                    self._redraw_status()
            finally:
                val = buffer.getvalue()
                # if val:
                #     print(val)
                #     exit(1)
                #     indented_output = "\n".join(
                #         f"    {line}" for line in val.splitlines()
                #     )
                #     with self.output_lock:
                #         self.test_results[test_name].append(f"\n{indented_output}\n")
                #         self._redraw_status()

        if hasattr(test_instance, "afterAll"):
            test_instance.afterAll()

        return bool(fails)

    def _format_success(self, method, duration):
        return f"    {GREEN_COLOR}✓ {method} ({duration * 1000:.2f}ms){RESET_COLOR}"

    def _format_failure(self, method, duration, traceback_output=""):
        failure_msg = (
            f"    {RED_COLOR}✗ {method} ({duration * 1000:.2f}ms){RESET_COLOR}"
        )
        if traceback_output:
            colored_traceback = "\n".join(
                f"  {CYAN_COLOR}{line}{RESET_COLOR}"
                for line in traceback_output.splitlines()
            )
            failure_msg += f"\n{colored_traceback}"
        return failure_msg

    def _redraw_status(self):
        self.clear_lines(self.printed_lines)  # Clear the previously printed lines
        self.output_lines = []  # Reset the output_lines for new drawing

        # Prepare and store the updated lines
        for test_name, results in self.test_results.items():
            self.output_lines.extend(results)

        # print(self.test_results)
        # # Print each new line and track how many lines were printed
        for line in self.output_lines:
            print(line)

        self.printed_lines = len(self.output_lines)  # Track how many lines were printed

    def _print_summary(self):
        duration = round(time.time() - self.start_time, 2)
        summary_lines = [
            f"\n{WHITE_BOLD}Test Summary{RESET_COLOR}\n",
            f"    Test Suites: {GREEN_COLOR}{self.passed_test_suites} passed{RESET_COLOR}, {RED_COLOR}{self.failed_test_suites} failed{RESET_COLOR}, {self.passed_test_suites + self.failed_test_suites} total",
            f"    Tests:       {GREEN_COLOR}{self.passed_tests} passed{RESET_COLOR}, {RED_COLOR}{self.failed_tests} failed{RESET_COLOR}, {self.passed_tests + self.failed_tests} total",
            f"    Time:        {duration}s\n",
        ]
        for line in summary_lines:
            print(line)

    def clear_lines(self, n=1):
        LINE_UP = "\033[1A"
        LINE_CLEAR = "\x1b[2K"
        for _ in range(n):
            print(LINE_UP, end=LINE_CLEAR)


class Stockfish:
    def __init__(
        self,
        prefix: List[str],
        path: str,
        args: List[str] = [],
        cli: bool = False,
    ):
        self.path = path
        self.process = None
        self.args = args
        self.cli = cli
        self.prefix = prefix
        self.output = []

        self.start()

    def start(self):
        if self.cli:
            self.process = subprocess.run(
                self.prefix + [self.path] + self.args,
                capture_output=True,
                text=True,
            )

            self.process.stdout

            return

        self.process = subprocess.Popen(
            self.prefix + [self.path] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

    def setoption(self, name: str, value: str):
        self.send_command(f"setoption name {name} value {value}")

    def send_command(self, command: str):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    @timeout_decorator(MAX_TIMEOUT)
    def equals(self, expected_output: str):
        for line in self.readline():
            if line == expected_output:
                return

    @timeout_decorator(MAX_TIMEOUT)
    def expect(self, expected_output: str):
        for line in self.readline():
            if fnmatch.fnmatch(line, expected_output):
                return

    @timeout_decorator(MAX_TIMEOUT)
    def contains(self, expected_output: str):
        for line in self.readline():
            if expected_output in line:
                return

    @timeout_decorator(MAX_TIMEOUT)
    def starts_with(self, expected_output: str):
        for line in self.readline():
            if line.startswith(expected_output):
                return

    @timeout_decorator(MAX_TIMEOUT)
    def check_output(self, callback):
        if not callback:
            raise ValueError("Callback function is required")

        for line in self.readline():
            if callback(line) == True:
                return

    def readline(self):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            line = self.process.stdout.readline().strip()
            self.output.append(line)

            yield line

    def clear_output(self):
        self.output = []

    def get_output(self) -> List[str]:
        return self.output

    def quit(self):
        self.send_command("quit")

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.stdout.close()
            return self.process.wait()

        return 0
