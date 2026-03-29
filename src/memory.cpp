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
#include <iostream>  // std::cerr

#if __has_include("features.h")
    #include <features.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <errno.h>
    #include <sys/mman.h>
    // IWYU pragma: no_include <bits/mman-map-flags-generic.h>
    #include <cstring>
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) \
  || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32)) \
  || defined(__e2k__)
    #define POSIXALIGNEDALLOC
    #include <stdlib.h>
#endif

#ifdef _WIN32
    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <ios>  // std::hex, std::dec
    #include <windows.h>

// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.

#endif


namespace Stockfish {

namespace {

struct AllocHeader {
    uint64_t magic;
    size_t   size;
    void*    base;
    bool     largePageAlloc;
};

constexpr uint64_t AllocHeaderMagic = 0x53544f434b464953ULL;

void* align_ptr_up(void* ptr, size_t alignment) {
    const auto ptr_value = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<void*>((ptr_value + alignment - 1) & ~(alignment - 1));
}

void* prepare_allocation(void* base, size_t size, size_t alignment, bool largePageAlloc) {
    if (!base)
        return nullptr;

    auto* userMem =
      static_cast<char*>(align_ptr_up(static_cast<char*>(base) + sizeof(AllocHeader), alignment));
    auto* header = reinterpret_cast<AllocHeader*>(userMem - sizeof(AllocHeader));

    *header = {AllocHeaderMagic, size, base, largePageAlloc};
    return userMem;
}

AllocHeader* header_from_user_ptr(void* mem) {
    auto* header = reinterpret_cast<AllocHeader*>(static_cast<char*>(mem) - sizeof(AllocHeader));
    assert(header->magic == AllocHeaderMagic);
    return header;
}

}  // namespace

// Wrappers for systems where the c++17 implementation does not guarantee the
// availability of aligned_alloc(). Memory allocated with std_aligned_alloc()
// must be freed with std_aligned_free().

void* std_aligned_alloc(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(alignment, size);
#elif defined(POSIXALIGNEDALLOC)
    void* mem = nullptr;
    posix_memalign(&mem, alignment, size);
    return mem;
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    return _mm_malloc(size, alignment);
#elif defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void std_aligned_free(void* ptr) {

#if defined(POSIXALIGNEDALLOC)
    free(ptr);
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    _mm_free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// aligned_large_pages_alloc() will return suitably aligned memory,
// if possible using large pages.

#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows([[maybe_unused]] size_t allocSize) {

    return windows_try_with_large_page_priviliges(
      [&](size_t largePageSize) {
          // Round up size to full pages and allocate
          allocSize = (allocSize + largePageSize - 1) & ~size_t(largePageSize - 1);
          return VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                              PAGE_READWRITE);
      },
      []() { return (void*) nullptr; });
}

void* aligned_large_pages_alloc_with_hint(size_t allocSize, bool) {
    constexpr size_t alignment = 4096;
    size_t           totalSize = allocSize + sizeof(AllocHeader) + alignment - 1;
    totalSize                  = (totalSize + alignment - 1) / alignment * alignment;

    // Try to allocate large pages
    void* mem            = aligned_large_pages_alloc_windows(totalSize);
    bool  largePageAlloc = mem != nullptr;

    // Fall back to regular, page-aligned, allocation if necessary
    if (!mem)
        mem = VirtualAlloc(nullptr, totalSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

    return prepare_allocation(mem, totalSize, alignment, largePageAlloc);
}

#else

    #if defined(__linux__) && defined(MAP_HUGE_SHIFT)
        #define HAS_HUGE_PAGES

static void* try_huge_pages_alloc(size_t allocSize, size_t alignment) {
    size_t size = allocSize + sizeof(AllocHeader) + alignment - 1;
    size        = (size + HugePageSize - 1) / HugePageSize * HugePageSize;
    void* mem   = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT), -1, 0);

    if (mem == MAP_FAILED)
        return nullptr;

    return prepare_allocation(mem, size, alignment, true);
}
    #endif  // defined(__linux__) && defined(MAP_HUGE_SHIFT)

void* aligned_large_pages_alloc_with_hint(size_t allocSize, [[maybe_unused]] bool hugePageHint) {
    #if defined(__linux__)
    constexpr size_t alignment = 2 * 1024 * 1024;  // 2MB page size assumed
    #else
    constexpr size_t alignment = 4096;  // small page size assumed
    #endif

    #ifdef HAS_HUGE_PAGES
    if (hugePageHint && allocSize >= HugePageSize)
    {
        void* mem = try_huge_pages_alloc(allocSize, alignment);
        if (mem)
            return mem;
    }
    #endif

    // Round up to multiples of alignment
    size_t size = allocSize + sizeof(AllocHeader) + alignment - 1;
    size        = (size + alignment - 1) / alignment * alignment;
    void* mem   = std_aligned_alloc(alignment, size);
    #if defined(MADV_HUGEPAGE)
    if (mem)
        madvise(mem, size, MADV_HUGEPAGE);
    #endif
    return prepare_allocation(mem, size, alignment, false);
}

#endif

void* aligned_large_pages_alloc(size_t size) {
    return aligned_large_pages_alloc_with_hint(size, false);
}

bool has_large_pages() {

#if defined(_WIN32)

    constexpr size_t page_size = 2 * 1024 * 1024;  // 2MB page size assumed
    void*            mem       = aligned_large_pages_alloc_windows(page_size);
    if (mem == nullptr)
    {
        return false;
    }
    else
    {
        VirtualFree(mem, 0, MEM_RELEASE);
        return true;
    }

#elif defined(__linux__)

    #if defined(MADV_HUGEPAGE)
    return true;
    #else
    return false;
    #endif

#else

    return false;

#endif
}


// aligned_large_pages_free() will free the previously memory allocated
// by aligned_large_pages_alloc(). The effect is a nop if mem == nullptr.

#if defined(_WIN32)

void aligned_large_pages_free(void* mem) {

    if (!mem)
        return;

    auto* header = header_from_user_ptr(mem);

    if (!VirtualFree(header->base, 0, MEM_RELEASE))
    {
        DWORD err = GetLastError();
        std::cerr << "Failed to free large page memory. Error code: 0x" << std::hex << err
                  << std::dec << std::endl;
        exit(EXIT_FAILURE);
    }
}

#else

void aligned_large_pages_free(void* mem) {
    if (!mem)
        return;

    auto* header = header_from_user_ptr(mem);

    if (!header->largePageAlloc)
    {
        std_aligned_free(header->base);
        return;
    }

    if (munmap(header->base, header->size) != 0)
    {
        std::cerr << "munmap failed: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif
}  // namespace Stockfish
