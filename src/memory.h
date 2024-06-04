/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#ifndef MEMORY_H_INCLUDED
#define MEMORY_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "types.h"

namespace Stockfish {

void* std_aligned_alloc(size_t alignment, size_t size);
void  std_aligned_free(void* ptr);
// memory aligned by page size, min alignment: 4096 bytes
void* aligned_large_pages_alloc(size_t size);
// nop if mem == nullptr
void aligned_large_pages_free(void* mem);

// frees memory which was placed there with placement new.
// works for both single objects and arrays of unknown bound
template<typename T, typename FREE_FUNC>
void memory_deleter(T* ptr, FREE_FUNC free_func) {
    if (!ptr)
        return;

    // Explicitly needed to call the destructor
    if constexpr (!std::is_trivially_destructible_v<T>)
        ptr->~T();

    free_func(ptr);
    return;
}

// frees memory which was placed there with placement new.
// works for both single objects and arrays of unknown bound
template<typename T, typename FREE_FUNC>
void memory_deleter_array(T* ptr, FREE_FUNC free_func) {
    if (!ptr)
        return;

    // Move back on the pointer to where the size is allocated.
    const size_t array_offset = std::max(sizeof(size_t), alignof(T));
    char*        raw_memory   = reinterpret_cast<char*>(ptr) - array_offset;

    if constexpr (!std::is_trivially_destructible_v<T>)
    {
        const size_t size = *reinterpret_cast<size_t*>(raw_memory);

        // Explicitly call the destructor for each element in reverse order
        for (size_t i = size; i-- > 0;)
            ptr[i].~T();
    }

    free_func(raw_memory);
}

// Allocates memory for a single object and places it there with placement new.
template<typename T, bool ZERO = true, typename ALLOC_FUNC, typename... Args>
inline std::enable_if_t<!std::is_array_v<T>, T*> memory_allocator(ALLOC_FUNC alloc_func,
                                                                  Args&&... args) {
    void* raw_memory = alloc_func(sizeof(T));
    ASSERT_ALIGNED(raw_memory, alignof(T));


    if constexpr (!ZERO)
    {
        static_assert(sizeof...(args) == 0,
                      "Cannot pass arguments when default initialization is requested.");

        // default initialization
        return new (raw_memory) T;
    }

    return new (raw_memory) T(std::forward<Args>(args)...);
}

// Allocates memory for an array of unknown bound and places it there with placement new.
template<typename T, bool ZERO = true, typename ALLOC_FUNC>
inline std::enable_if_t<std::is_array_v<T>, std::remove_extent_t<T>*>
memory_allocator(ALLOC_FUNC alloc_func, size_t num) {
    using ElementType = std::remove_extent_t<T>;

    const size_t array_offset = std::max(sizeof(size_t), alignof(ElementType));

    // save the array size in the memory location
    char* raw_memory =
      reinterpret_cast<char*>(alloc_func(array_offset + num * sizeof(ElementType)));
    ASSERT_ALIGNED(raw_memory, alignof(T));

    new (raw_memory) size_t(num);

    for (size_t i = 0; i < num; ++i)
    {
        if constexpr (ZERO)
            new (raw_memory + array_offset + i * sizeof(ElementType)) ElementType();

        // default initialization
        new (raw_memory + array_offset + i * sizeof(ElementType)) ElementType;
    }

    // Need to return the pointer at the start of the array so that the indexing in unique_ptr<T[]> works
    return reinterpret_cast<ElementType*>(raw_memory + array_offset);
}

//
//
// aligned large page unique ptr
//
//

template<typename T>
struct LargePageDeleter {
    void operator()(T* ptr) const { return memory_deleter<T>(ptr, aligned_large_pages_free); }
};

template<typename T>
struct LargePageArrayDeleter {
    void operator()(T* ptr) const { return memory_deleter_array<T>(ptr, aligned_large_pages_free); }
};

template<typename T>
using LargePagePtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, LargePageArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, LargePageDeleter<T>>>;


class LargePageAllocator {
   public:
    template<typename T, typename... Args>
    static LargePagePtr<T> make_unique(Args&&... args) {
        return make_unique_large_page_impl<T, true>(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    static LargePagePtr<T> make_unique_for_overwrite(Args&&... args) {
        return make_unique_large_page_impl<T, false>(std::forward<Args>(args)...);
    }

   private:
    template<typename T, bool ZERO = true, typename... Args>
    static LargePagePtr<T> make_unique_large_page_impl(Args&&... args) {
        using ElementType = std::remove_extent_t<T>;

        static_assert(
          alignof(ElementType) <= 4096,
          "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");

        if constexpr (!std::is_array_v<T>)
        {
            T* memory =
              memory_allocator<T, ZERO>(aligned_large_pages_alloc, std::forward<Args>(args)...);

            return LargePagePtr<T>(memory);
        }

        ElementType* memory =
          memory_allocator<T, ZERO>(aligned_large_pages_alloc, std::forward<Args>(args)...);

        return LargePagePtr<T>(memory);
    }
};

//
//
// aligned unique ptr
//
//

template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const { return memory_deleter<T>(ptr, std_aligned_free); }
};

template<typename T>
struct AlignedArrayDeleter {
    void operator()(T* ptr) const { return memory_deleter_array<T>(ptr, std_aligned_free); }
};

template<typename T>
using AlignedPtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, AlignedArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, AlignedDeleter<T>>>;

class AlignedAllocator {
   public:
    template<typename T, typename... Args>
    static AlignedPtr<T> make_unique(Args&&... args) {
        return AlignedAllocator::make_unique_impl<T, true>(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    static AlignedPtr<T> make_unique_for_overwrite(Args&&... args) {
        return AlignedAllocator::make_unique_impl<T, false>(std::forward<Args>(args)...);
    }

   private:
    template<typename T, bool ZERO = true, typename... Args>
    static AlignedPtr<T> make_unique_impl(Args&&... args) {
        using ElementType = std::remove_extent_t<T>;

        if constexpr (!std::is_array_v<T>)
        {
            const auto func   = [](size_t size) { return std_aligned_alloc(alignof(T), size); };
            T*         memory = memory_allocator<T, ZERO>(func, std::forward<Args>(args)...);

            return AlignedPtr<T>(memory);
        }

        const auto func = [](size_t size) { return std_aligned_alloc(alignof(ElementType), size); };
        ElementType* memory = memory_allocator<T, ZERO>(func, std::forward<Args>(args)...);

        return AlignedPtr<T>(memory);
    }
};


// Get the first aligned element of an array.
// ptr must point to an array of size at least `sizeof(T) * N + alignment` bytes,
// where N is the number of elements in the array.
template<uintptr_t Alignment, typename T>
T* align_ptr_up(T* ptr) {
    static_assert(alignof(T) < Alignment);

    const uintptr_t ptrint = reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(ptr));
    return reinterpret_cast<T*>(
      reinterpret_cast<char*>((ptrint + (Alignment - 1)) / Alignment * Alignment));
}


}  // namespace Stockfish

#endif  // #ifndef MEMORY_H_INCLUDED
