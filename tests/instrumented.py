import subprocess
from typing import List, Optional
import argparse
import os


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
                """
race:Stockfish::TTEntry::read
race:Stockfish::TTEntry::save

race:Stockfish::TranspositionTable::probe
race:Stockfish::TranspositionTable::hashfull
"""
            )

        os.environ["TSAN_OPTIONS"] = "suppressions=./tsan.supp"


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


class Stockfish:
    def __init__(self, stockfish_path: str, args: List[str]):
        self.stockfish_path = stockfish_path
        self.process = None
        self.args = args
        self.output = []

        self.start()

    def __get_prefix(self):
        if args.valgrind:
            return Valgrind.get_valgrind_command()
        if args.valgrind_thread:
            return Valgrind.get_valgrind_thread_command()

        return ""

    def start(self):
        self.process = subprocess.Popen(
            self.__get_prefix() + [self.stockfish_path] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

    def setoption(self, name: str, value: str):
        """Set an option in Stockfish."""
        self.send_command(f"setoption name {name} value {value}")

    def send_command(self, command: str):
        """Send a command to Stockfish without waiting for output."""
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def expect(self, expected_output: str) -> Optional[str]:
        """Wait for expected output from Stockfish."""
        if not self.process:
            raise RuntimeError("Stockfish process is not started")

        while True:
            output_line = self.process.stdout.readline().strip()

            if not output_line:
                break

            self.output.append(output_line)

            if output_line == expected_output:
                return True

        return False

    def check_output(self, callback) -> Optional[str]:
        """Explicitly check for output after commands have been sent."""
        if not self.process:
            return None

        if not callback:
            raise ValueError("Callback function is required")

        while True:
            output_line = self.process.stdout.readline().strip()

            if not output_line:
                break

            self.output.append(output_line)

            if callback(output_line) == True:
                return True

        return None

    def get_output(self) -> List[str]:
        return self.output

    def close(self):
        if self.process:
            # self.process.stdin.close()
            # self.process.stdout.close()
            # self.process.stderr.close()
            # self.process.terminate()
            return self.process.wait()

        return 0


class TestStockfishCLI:

    @staticmethod
    def test():

        for args in [
            "eval",
            "go nodes 1000",
            "go depth 10",
            "go perft 4",
            "go movetime 1000",
            "go wtime 8000 btime 8000 winc 500 binc 500",
            "go wtime 1000 btime 1000 winc 0 binc 0",
            "go wtime 1000 btime 1000 winc 0 binc 0",
            "go wtime 1000 btime 1000 winc 0 binc 0 movestogo 5",
            "go movetime 200",
            "go nodes 20000 searchmoves e2e4 d2d4",
            "bench 128 $threads 8 default depth",
            "bench 128 $threads 3 bench_tmp.epd depth",
            "export_net verify.nnue",
            "d",
            "compiler",
            "license",
            "uci",
        ]:
            stockfish = Stockfish("../src/stockfish", args.split(" "))
            stockfish.send_command(args)
            # stockfish.expect(
            #     "Final evaluation       +0.09 (white side) [with scaled NNUE, ...]"
            # )

            print(stockfish.close())
            assert stockfish.close() == 0


class TestStockfish:
    def setUp(self):
        # Replace with actual path to Stockfish binary
        self.stockfish = Stockfish("../src/stockfish")

        if args.valgrind_thread or args.sanitizer_thread:
            self.stockfish.setoption("Threads", "2")
            TSAN.set_tsan_option()

    def tearDown(self):
        self.stockfish.close()

    def test_is_ready(self):
        self.stockfish.send_command("isready")

        output = self.stockfish.expect("readyok")

        assert output == True


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stockfish with testing options")
    parser.add_argument("--valgrind", action="store_true", help="Run valgrind testing")
    parser.add_argument(
        "--valgrind-thread", action="store_true", help="Run valgrind-thread testing"
    )
    parser.add_argument(
        "--sanitizer-undefined",
        action="store_true",
        help="Run sanitizer-undefined testing",
    )
    parser.add_argument(
        "--sanitizer-thread", action="store_true", help="Run sanitizer-thread testing"
    )

    return parser.parse_args()


# To run the tests
if __name__ == "__main__":
    args = parse_args()

    TestStockfishCLI.test()
    # test_stockfish = TestStockfish()

    # test_stockfish.setUp()

    # test_stockfish.test_is_ready()

    # test_stockfish.tearDown()
