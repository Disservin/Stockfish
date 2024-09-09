import subprocess
from typing import List, Optional, Callable, Any
import argparse
import os
import re
import collections
import time
import sys
import traceback
import fnmatch
import signal
from functools import wraps
from contextlib import redirect_stdout
import io


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
    def __init__(self, message, timeout):
        self.message = message
        self.timeout = timeout


def timeout(seconds: int) -> Callable:
    def decorator(func: Callable) -> Callable:
        def _handle_timeout(signum, frame):
            raise TimeoutException(
                f"Function '{func.__name__}' timed out after {seconds} seconds",
                seconds,
            )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm after function completion
                signal.alarm(0)
            return result

        return wrapper

    return decorator


class MiniTestFramework:
    def __init__(self):
        self.passed_test_suites = 0
        self.failed_test_suites = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0

    def has_failed(self):
        return bool(self.failed_test_suites)

    def run(self, classes: List[type]):
        self.start_time = time.time()

        for test_class in classes:
            ret = self.__run(test_class)
            if ret:
                self.failed_test_suites += 1
            else:
                self.passed_test_suites += 1

        duration = round(time.time() - self.start_time, 2)

        print("\n\033[1mTest Summary\033[0m\n")
        print(
            f"    Test Suites: \033[42m{self.passed_test_suites} passed\033[0m, {self.failed_test_suites} total"
        )
        print(
            f"    Tests:       \033[42m{self.passed_tests} passed\033[0m, {self.failed_tests} failed, {self.total_tests} total"
        )
        print(f"    Time:        {round(duration, 2)}s\n")

        return bool(self.failed_test_suites)

    def __run(self, test_class):
        test_instance = test_class()
        methods = [
            method for method in test_instance.__ordered__ if method.startswith("test_")
        ]

        if "beforeAll" in dir(test_instance):
            test_instance.beforeAll()

        print(f"\nTest Suite: {test_instance.__class__.__name__}")

        fails = 0

        for method in methods:
            self.total_tests += 1
            print(f"    Running {method}... ", end="", flush=True)
            buffer = io.StringIO()

            try:
                t0 = time.time()

                # Redirect stdout to buffer for capturing test output
                with redirect_stdout(buffer):
                    getattr(test_instance, method)()

                    if hasattr(test_instance, "afterEach"):
                        test_instance.afterEach()

                duration = time.time() - t0

                print(
                    f"\033[32m✓\033[0m \033[42m\033[97m PASS \033[0m  ({duration * 1000:.2f}ms)"
                )

                self.passed_tests += 1
            except TimeoutException as e:
                print(
                    f"\033[31m✗\033[0m \033[41m\033[97m FAIL \033[0m  hit execution limit of {e.timeout} seconds"
                )

                fails += 1
                self.failed_tests += 1

            except AssertionError:
                duration = time.time() - t0
                print(
                    f"\033[31m✗\033[0m \033[41m\033[97m FAIL \033[0m  ({duration * 1000:.2f}ms)"
                )

                traceback_output = "".join(traceback.format_tb(sys.exc_info()[2]))

                color_code = "\033[36m"
                reset_code = "\033[0m"

                colored_traceback = "\n".join(
                    f"  {color_code}{line}{reset_code}"
                    for line in traceback_output.splitlines()
                )

                print(colored_traceback)

                fails += 1
                self.failed_tests += 1
            finally:
                val = buffer.getvalue()

                if val:
                    indented_output = "\n".join(
                        f"    {line}" for line in val.splitlines()
                    )
                    print(f"\033[2m{indented_output}\033[0m")

        if "afterAll" in dir(test_instance):
            test_instance.afterAll()

        return bool(fails)


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
        with open("tsan.supp", "w") as f:
            f.write(
                ""
                #                 """
                # race:Stockfish::TTEntry::read
                # race:Stockfish::TTEntry::save
                # race:Stockfish::TranspositionTable::probe
                # race:Stockfish::TranspositionTable::hashfull
                # """
            )

        os.environ["TSAN_OPTIONS"] = "suppressions=./tsan.supp"

    @staticmethod
    def unset_tsan_option():
        os.environ.pop("TSAN_OPTIONS", None)

        os.remove("tsan.supp")


class EPD:
    @staticmethod
    def create_bench_epd():
        with open("bench_tmp.epd", "w") as f:
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
        os.remove("bench_tmp.epd")


class Stockfish:
    def __init__(
        self,
        prefix: List[str],
        stockfish_path: str,
        args: List[str] = [],
        cli: bool = False,
    ):
        self.stockfish_path = stockfish_path
        self.process = None
        self.args = args
        self.cli = cli
        self.prefix = prefix
        self.output = []

        self.start()

    def start(self):
        if self.cli:
            self.process = subprocess.run(
                self.prefix + [self.stockfish_path] + self.args,
                capture_output=True,
                text=True,
            )

            self.process.stdout

            return

        self.process = subprocess.Popen(
            self.prefix + [self.stockfish_path] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
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

    @timeout(60 * 5)
    def equals(self, expected_output: str):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            output_line = self.readline()

            if output_line == expected_output:
                return

    @timeout(60 * 5)
    def expect(self, expected_output: str):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            output_line = self.readline()

            if fnmatch.fnmatch(output_line, expected_output):
                return

    @timeout(60 * 5)
    def contains(self, expected_output: str):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            output_line = self.readline()

            if expected_output in output_line:
                return

    @timeout(60 * 5)
    def starts_with(self, expected_output: str):
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            output_line = self.readline()

            if output_line.startswith(expected_output):
                return

    @timeout(60 * 5)
    def check_output(self, callback):
        if not self.process:
            return None

        if not callback:
            raise ValueError("Callback function is required")

        while True:
            output_line = self.readline()

            if callback(output_line) == True:
                return

    def readline(self) -> str:
        if not self.process:
            return None

        line = self.process.stdout.readline().strip()
        self.output.append(line)

        return line

    def get_output(self) -> List[str]:
        return self.output

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.terminate()
            return self.process.wait()

        return 0
