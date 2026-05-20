/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "memory.h"

#include <cstdlib>
#include <new>

#if defined(__linux__) && !defined(__ANDROID__)
    #include <sys/mman.h>
#endif

#ifdef _WIN32
    #include <ios>
    #include <iostream>
    #include <ostream>
    #include <windows.h>
#endif

namespace Stockfish {

namespace {

constexpr size_t round_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

void* std_aligned_alloc(size_t alignment, size_t size) {
    return ::operator new(size, std::align_val_t(alignment), std::nothrow);
}

void std_aligned_free(void* ptr, size_t alignment) noexcept {
    ::operator delete(ptr, std::align_val_t(alignment));
}

#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows(size_t allocSize) {
    return windows_try_with_large_page_priviliges(
      [&](size_t largePageSize) {
          // Round up size to full pages and allocate
          allocSize = round_up(allocSize, largePageSize);
          return VirtualAlloc(nullptr, allocSize,
                              MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                              PAGE_READWRITE);
      },
      [] { return static_cast<void*>(nullptr); });
}

void* aligned_large_pages_alloc(size_t allocSize) {
    if (void* mem = aligned_large_pages_alloc_windows(allocSize))
        return mem;

    return VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void aligned_large_pages_free(void* mem) {
    if (mem && !VirtualFree(mem, 0, MEM_RELEASE))
    {
        DWORD err = GetLastError();
        std::cerr << "Failed to free large page memory. Error code: 0x"
                  << std::hex << err << std::dec << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#else

void* aligned_large_pages_alloc(size_t allocSize) {
#if defined(__linux__)
    constexpr size_t Alignment = 2 * 1024 * 1024;
#else
    constexpr size_t Alignment = 4096;
#endif

    const size_t size = round_up(allocSize, Alignment);
    void* mem = std_aligned_alloc(Alignment, size);

#if defined(MADV_HUGEPAGE)
    if (mem)
        madvise(mem, size, MADV_HUGEPAGE);
#endif

    return mem;
}

void aligned_large_pages_free(void* mem) {
#if defined(__linux__)
    constexpr size_t Alignment = 2 * 1024 * 1024;
#else
    constexpr size_t Alignment = 4096;
#endif

    std_aligned_free(mem, Alignment);
}

#endif

bool has_large_pages() {
#if defined(_WIN32)
    constexpr size_t PageSize = 2 * 1024 * 1024;

    void* mem = aligned_large_pages_alloc_windows(PageSize);
    if (!mem)
        return false;

    aligned_large_pages_free(mem);
    return true;

#elif defined(__linux__) && defined(MADV_HUGEPAGE)
    return true;
#else
    return false;
#endif
}

} // namespace Stockfish