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

#ifndef MEMORY_ALLOC_H_INCLUDED
#define MEMORY_ALLOC_H_INCLUDED

#include <algorithm>
#include <cstddef>
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
template<typename T, typename ALLOC_FUNC, typename... Args>
inline std::enable_if_t<!std::is_array_v<T>, T*> memory_allocator(ALLOC_FUNC alloc_func,
                                                                  Args&&... args) {
    void* raw_memory = alloc_func(sizeof(T));
    ASSERT_ALIGNED(raw_memory, alignof(T));
    return new (raw_memory) T(std::forward<Args>(args)...);
}

// Allocates memory for an array of unknown bound and places it there with placement new.
template<typename T, typename ALLOC_FUNC>
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
        new (raw_memory + array_offset + i * sizeof(ElementType)) ElementType();

    // Need to return the pointer at the start of the array so that the indexing in unique_ptr<T[]> works
    return reinterpret_cast<ElementType*>(raw_memory + array_offset);
}

//
//
// aligned large page unique ptr
//
//

template<typename T>
struct AlignedLargePageDeleter {
    void operator()(T* ptr) const { return memory_deleter<T>(ptr, aligned_large_pages_free); }
};

template<typename T>
struct AlignedLargePageArrayDeleter {
    void operator()(T* ptr) const { return memory_deleter_array<T>(ptr, aligned_large_pages_free); }
};

// make_unique_aligned_large_page for single objects
template<typename T, typename... Args>
inline std::enable_if_t<!std::is_array_v<T>, std::unique_ptr<T, AlignedLargePageDeleter<T>>>
make_unique_aligned_large_page(Args&&... args) {
    static_assert(alignof(T) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");
    T* obj = memory_allocator<T>(aligned_large_pages_alloc, std::forward<Args>(args)...);
    return std::unique_ptr<T, AlignedLargePageDeleter<T>>(obj);
}

// make_unique_aligned_large_page for arrays of unknown bound
template<typename T>
inline std::enable_if_t<std::is_array_v<T>,
                        std::unique_ptr<T, AlignedLargePageArrayDeleter<std::remove_extent_t<T>>>>
make_unique_aligned_large_page(size_t num) {
    using ElementType = std::remove_extent_t<T>;

    static_assert(alignof(ElementType) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");

    ElementType* memory = memory_allocator<T>(aligned_large_pages_alloc, num);

    return std::unique_ptr<T, AlignedLargePageArrayDeleter<ElementType>>(memory);
}

template<typename T>
using AlignedLargePageUniquePtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, AlignedLargePageArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, AlignedLargePageDeleter<T>>>;

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

// make_unique_aligned for single objects
template<typename T, typename... Args>
inline std::enable_if_t<!std::is_array_v<T>, std::unique_ptr<T, AlignedDeleter<T>>>
make_unique_aligned(Args&&... args) {
    T* obj = memory_allocator<T>([](size_t size) { return std_aligned_alloc(alignof(T), size); },
                                 std::forward<Args>(args)...);
    return std::unique_ptr<T, AlignedDeleter<T>>(obj);
}

// make_unique_aligned for arrays of unknown bound
template<typename T>
inline std::enable_if_t<std::is_array_v<T>,
                        std::unique_ptr<T, AlignedArrayDeleter<std::remove_extent_t<T>>>>
make_unique_aligned(size_t num) {
    using ElementType = std::remove_extent_t<T>;

    ElementType* memory = memory_allocator<T>(
      [](size_t size) { return std_aligned_alloc(alignof(ElementType), size); }, num);

    return std::unique_ptr<T, AlignedArrayDeleter<ElementType>>(memory);
}

template<typename T>
using AlignedUniquePtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, AlignedArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, AlignedDeleter<T>>>;

}  // namespace Stockfish

#endif  // #ifndef MEMORY_ALLOC_H_INCLUDED
