#!/usr/bin/env python3

import json


def generate_build_matrix():
    configs = [
        {
            "name": "Ubuntu 22.04 GCC",
            "os": "ubuntu-22.04",
            "simple_name": "ubuntu",
            "compiler": "g++",
            "comp": "gcc",
            "shell": "bash",
            "archive_ext": "tar",
            "sde": "/home/runner/work/Stockfish/Stockfish/.output/sde-temp-files/sde-external-9.27.0-2023-09-13-lin/sde -future --",
        },
        {
            "name": "Ubuntu 22.04 ARM GCC",
            "os": "ubuntu-22.04-arm",
            "simple_name": "ubuntu_arm",
            "compiler": "g++",
            "comp": "gcc",
            "shell": "bash",
            "archive_ext": "tar",
        },
        {
            "name": "MacOS 13 Apple Clang",
            "os": "macos-13",
            "simple_name": "macos",
            "compiler": "clang++",
            "comp": "clang",
            "shell": "bash",
            "archive_ext": "tar",
        },
        {
            "name": "MacOS 14 Apple Clang M1",
            "os": "macos-14",
            "simple_name": "macos-m1",
            "compiler": "clang++",
            "comp": "clang",
            "shell": "bash",
            "archive_ext": "tar",
        },
        {
            "name": "Windows 2022 Mingw-w64 GCC x86_64",
            "os": "windows-2022",
            "simple_name": "windows",
            "compiler": "g++",
            "comp": "mingw",
            "msys_sys": "mingw64",
            "msys_env": "x86_64-gcc",
            "shell": "msys2 {0}",
            "ext": ".exe",
            "sde": "/d/a/Stockfish/Stockfish/.output/sde-temp-files/sde-external-9.27.0-2023-09-13-win/sde.exe -future --",
            "archive_ext": "zip",
        },
    ]

    binaries = [
        "x86-64",
        "x86-64-sse41-popcnt",
        "x86-64-avx2",
        "x86-64-bmi2",
        "x86-64-avxvnni",
        "x86-64-avx512",
        "x86-64-vnni256",
        "x86-64-vnni512",
        "apple-silicon",
        "armv7",
        "armv7-neon",
        "armv8",
        "armv8-dotprod",
    ]

    compatibility = {
        "ubuntu-22.04": [
            "x86-64",
            "x86-64-sse41-popcnt",
            "x86-64-avx2",
            "x86-64-bmi2",
            "x86-64-avx512",
            "x86-64-vnni256",
            "x86-64-vnni512",
            # Note: x86-64-avxvnni removed as it was causing issues
        ],
        "ubuntu-22.04-arm": ["armv7", "armv7-neon", "armv8", "armv8-dotprod"],
        "macos-13": ["x86-64", "x86-64-sse41-popcnt", "x86-64-avx2", "x86-64-bmi2"],
        "macos-14": ["apple-silicon"],
        "windows-2022": [
            "x86-64",
            "x86-64-sse41-popcnt",
            "x86-64-avx2",
            "x86-64-bmi2",
            "x86-64-avxvnni",
            "x86-64-avx512",
            "x86-64-vnni256",
            "x86-64-vnni512",
        ],
    }

    exclude = []
    for os_name, compatible_binaries in compatibility.items():
        incompatible_binaries = [b for b in binaries if b not in compatible_binaries]

        # add exclusions
        for binary in incompatible_binaries:
            exclude.append({"binaries": binary, "config": {"os": os_name}})

    matrix = {"config": configs, "binaries": binaries, "exclude": exclude}

    return matrix


def print_compatibility_table(matrix):
    """Prints a simple compatibility table showing which binaries can be built on which OS."""
    configs = matrix["config"]
    binaries = matrix["binaries"]
    exclusions = matrix["exclude"]

    os_excluded_binaries = {}
    for exclusion in exclusions:
        if "os" in exclusion["config"]:
            os_name = exclusion["config"]["os"]
            binary = exclusion["binaries"]
            if os_name not in os_excluded_binaries:
                os_excluded_binaries[os_name] = set()
            os_excluded_binaries[os_name].add(binary)

    os_column_width = 20
    binary_column_width = 15
    total_width = os_column_width + (binary_column_width * len(binaries))

    print("\nCompatibility Matrix:")
    print("-" * total_width)
    header = "OS".ljust(os_column_width)
    for binary in binaries:
        header += binary.ljust(binary_column_width)
    print(header)
    print("-" * total_width)

    for config in configs:
        os_name = config["os"]
        row = os_name.ljust(os_column_width)

        excluded = os_excluded_binaries.get(os_name, set())
        for binary in binaries:
            if binary in excluded:
                row += "❌".ljust(binary_column_width)
            else:
                row += "✅".ljust(binary_column_width)
        print(row)

    print("-" * total_width)


if __name__ == "__main__":
    matrix = generate_build_matrix()

    print_compatibility_table(matrix)

    with open(".github/ci/matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
