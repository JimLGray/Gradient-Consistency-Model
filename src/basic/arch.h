//
//  arch.hpp
//  sift_depth
//
//  Created by Aous Naman on 4/5/17.
//  Copyright © 2017 Aous Naman. All rights reserved.
//

#ifndef SD_ARCH_H
#define SD_ARCH_H

#include <stdio.h>
#include <cstdint>

namespace sd {

  ////////////////////////////////////////////////////////////////////////////
  // constants
  ////////////////////////////////////////////////////////////////////////////
  const int byte_alignment = 32;
  const int log_byte_alignment = 5;

  /////////////////////////////////////////////////////////////////////////////
  // templates for alignment
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // finds the size such that it is a mulitple of byte_alignment
  template <typename T>
  size_t calc_aligned_size(size_t size) {
    size = size * sizeof(T) + byte_alignment - 1;
    size &= ~((1ULL << log_byte_alignment) - 1);
    size /= sizeof(T);
    return size;
  }

  /////////////////////////////////////////////////////////////////////////////
  // moves the pointer to first address that is a multiple of byte_alignment
  template <typename T>
  inline T *align_ptr(T *ptr) {
    intptr_t p = reinterpret_cast<intptr_t>(ptr);
    p += byte_alignment - 1;
    p &= ~((1ULL << log_byte_alignment) - 1);
    return reinterpret_cast<T *>(p);
  }

  /////////////////////////////////////////////////////////////////////////////
  //                             cpu features
  /////////////////////////////////////////////////////////////////////////////
  bool is_mmx_available();
  bool is_sse_available();
  bool is_sse2_available();
  bool is_sse3_available();
  bool is_ssse3_available();
  bool is_sse41_available();
  bool is_sse42_available();
  bool is_avx_available();
  bool is_avx2_available();
  bool is_avx2fma_available();
  bool is_avx512f_available();
  bool is_avx512cd_available();
  bool is_avx512dq_available();
  bool is_avx512bw_available();
  bool is_avx512ifma_available();
  bool is_avx512vbmi_available();

  /////////////////////////////////////////////////////////////////////////////
  //                               types
  /////////////////////////////////////////////////////////////////////////////
  typedef uint8_t ui8;
  typedef int8_t si8;
  typedef uint16_t ui16;
  typedef int16_t si16;
  typedef uint32_t ui32;
  typedef int32_t si32;
  typedef uint64_t ui64;
  typedef int64_t si64;
  typedef float flt32;
  typedef double flt64;


}

#endif // SD_ARCH_H
