//
//  arch.cpp
//  sift_depth
//
//  Created by Aous Naman on 4/5/17.
//  Copyright © 2017 Aous Naman. All rights reserved.
//

#include <immintrin.h>
#include <cstdint>
#include <cassert>
#include "arch.h"

//////////////////////////////////////////////////////////////////////////////
// This snippet is borrowed from Intel; I forgot from where exactly
bool run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
  __cpuidex((int*)abcd, eax, ecx);
#else
  uint32_t ebx = 0, edx = 0;
#if defined( __i386__ ) && defined ( __PIC__ )
  /* in case of PIC under 32-bit EBX cannot be clobbered */
  __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
           "+a" (eax), "+c" (ecx), "=d" (edx) );
#else
  __asm__ ( "cpuid" : "+b" (ebx), "+a" (eax), "+c" (ecx), "=d" (edx) );
#endif
  abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
  return true;
}

//////////////////////////////////////////////////////////////////////////////
uint64_t read_xcr(uint32_t index)
{
#if defined(_MSC_VER)
  return _xgetbv(index);
#else
  uint32_t eax = 0, edx = 0;
  __asm__ ( "xgetbv" : "=a" (eax), "=d" (edx) : "c" (index) );
  return ((uint64_t)edx << 32) | eax;
#endif
}

////////////////////////////////////////////////////////////////////////////
// statics;
static uint32_t mmx_abcd[4];
static bool mmx_initialized = run_cpuid(1, 0, mmx_abcd);
static bool mmx_avail = ((mmx_abcd[3] & 0x00800000) == 0x00800000);
static bool sse_avail = ((mmx_abcd[3] & 0x02000000) == 0x02000000);
static bool sse2_avail = ((mmx_abcd[3] & 0x04000000) == 0x04000000);
static bool sse3_avail = ((mmx_abcd[2] & 0x00000001) == 0x00000001);
static bool ssse3_avail = ((mmx_abcd[2] & 0x00000200) == 0x00000200);
static bool sse41_avail = ((mmx_abcd[2] & 0x00080000) == 0x00080000);
static bool sse42_avail = ((mmx_abcd[2] & 0x00100000) == 0x00080000);
static bool osxsave_avail = ((mmx_abcd[2] & 0x08000000) == 0x08000000);
static uint64_t xcr_val = read_xcr(0);
static bool ymm_avail = osxsave_avail && ((xcr_val & 0x6) == 0x6);
static bool avx_avail = ymm_avail && (mmx_abcd[2] & 0x10000000);
static uint32_t avx2_abcd[4];
static bool avx2_initialized = run_cpuid(7, 0, avx2_abcd);
static bool avx2_avail = avx2_abcd[1] & 0x20;
static bool avx2fma_avail = avx2_avail && ((mmx_abcd[2] & 0x1000) == 0x1000);
static bool zmm_avail = osxsave_avail && ((xcr_val & 0xE) == 0xE);
static bool avx512vl_avail = avx2_abcd[1] & 0x80000000;
static bool avx512_avail = zmm_avail && avx512vl_avail;
static bool avx512f_avail = avx512_avail && (avx2_abcd[1] & 0x10000);
static bool avx512cd_avail = avx512_avail && (avx2_abcd[1] & 0x10000000);
static bool avx512dq_avail = avx512_avail && (avx2_abcd[1] & 0x20000);
static bool avx512bw_avail = avx512_avail && (avx2_abcd[1] & 0x40000000);
static bool avx512ifma_avail = avx512_avail && (avx2_abcd[1] & 0x200000);
static bool avx512vbmi_avail = avx512_avail && (avx2_abcd[2] & 0x2);


namespace sd {

  ////////////////////////////////////////////////////////////////////////////
  bool is_mmx_available() {
    assert(mmx_initialized);
    return mmx_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_sse_available() {
    assert(mmx_initialized);
    return sse_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_sse2_available() {
    assert(mmx_initialized);
    return sse2_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_sse3_available() {
    assert(mmx_initialized);
    return sse3_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_ssse3_available() {
    assert(mmx_initialized);
    return ssse3_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_sse41_available() {
    assert(mmx_initialized);
    return sse41_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_sse42_available() {
    assert(mmx_initialized);
    return sse42_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx_available() {
    assert(mmx_initialized);
    return avx_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx2_available() {
    assert(avx2_initialized);
    return avx2_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx2fma_available() {
    assert(avx2_initialized);
    return avx2fma_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512f_available() {
    assert(avx2_initialized);
    return avx512f_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512cd_available() {
    assert(avx2_initialized);
    return avx512cd_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512dq_available() {
    assert(avx2_initialized);
    return avx512dq_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512bw_available() {
    assert(avx2_initialized);
    return avx512bw_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512ifma_available() {
    assert(avx2_initialized);
    return avx512ifma_avail;
  }

  ////////////////////////////////////////////////////////////////////////////
  bool is_avx512vbmi_available() {
    assert(avx2_initialized);
    return avx512vbmi_avail;
  }
}
