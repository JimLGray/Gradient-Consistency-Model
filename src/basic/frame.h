//
//  frame.hpp
//  sift_depth
//
//  Created by Aous Naman on 4/5/17.
//  Copyright © 2017 Aous Naman. All rights reserved.
//

#ifndef SD_FRAME_H
#define SD_FRAME_H

#include <cstdio>
#include <cassert>
#include <cstring>
#include "arch.h"

namespace sd {

  //////////////////////////////////////////////////////////////////////////////
  struct loc
  { int x, y; loc(int x = 0, int y = 0) : x(x), y(y) {} };
  struct floc
  { sd::flt32 x, y; floc(sd::flt32 x=0.0f, sd::flt32 y=0.0f) : x(x), y(y) {} };

  /////////////////////////////////////////////////////////////////////////////
  // stores a frame with interleaved channels
  // the structure is reusable and can grow if needed
  // stride is a multple of byte_alignment, and data is always aligned
  template<typename T>
  class interleaved_frame {
  public:
    interleaved_frame() {
      store_size = ext = width = height = stride = num_ch = 0;
      store = data = NULL;
    }
    interleaved_frame(int width, int height, int num_channels,
                      int extension)
    {
      this->store_size = this->ext = this->width = this->height = 0;
      this->stride = this->num_ch = 0;
      this->store = this->data = NULL;
      init(width, height, num_channels, extension);
    }
    interleaved_frame(int width, int height, int num_channels, int extension, T* store)
    {
      this->ext = extension;
      this->width = width;
      this->height = height;
      this->stride = (int)((width + 2 * ext) * num_ch);
      size_t size = calc_aligned_size<T>((size_t)stride * (height + 2*ext));
      this->num_ch = num_channels;
      this->store_size = size;
      this->store = NULL;
      this->data = store + ext * stride + ext;
    }
    ~interleaved_frame() { if (store) delete[] store; }
    T* get_row(int row) { return data + row * stride; }
    const T* get_row(int row) const { return data + row * stride; }
    int get_ext() const { return ext; }
    int get_width() const { return width; }
    int get_height() const { return height; }
    int get_stride() const { return stride; }
    int get_num_ch() const { return num_ch; }
    void init(int width, int height, int num_channels, int extension)
    {
      this->width = width;
      this->height = height;
      this->ext = extension;
      this->num_ch = num_channels;
      this->stride = (int)calc_aligned_size<T>((width + 2 * ext) * num_ch);
      size_t size = calc_aligned_size<T>((size_t)stride * (height + 2*ext));
      if (store_size < size) {
        if (store) delete[] store;
        store = new T[size];
        store_size = size;
      }
      data = store + ext * stride + ext;
    }
    void zohold_extend()
    {
      memcpy(get_row(-1), get_row(0), width*num_ch*sizeof(T));
      memcpy(get_row(height), get_row(height-1), width*num_ch*sizeof(T));
      for (int y = -1; y <= height; ++y)
      {
        T *ldp = get_row(y) - num_ch, *lsp = get_row(y);
        T *rdp = get_row(y)+width*num_ch, *rsp=get_row(y)+(width-1)*num_ch;
        for (int x = 0; x < num_ch; ++x)
        { *ldp++ = *lsp++; *rdp++ = *rsp++; }
      }
    }
    void sym_extend()
    {
      memcpy(get_row(-1), get_row(1), width*num_ch*sizeof(T));
      memcpy(get_row(height), get_row(height-2), width*num_ch*sizeof(T));
      for (int y = -1; y <= height; ++y)
      {
        T *ldp = get_row(y) - num_ch, *lsp = get_row(y) + num_ch;
        T *rdp = get_row(y)+width*num_ch, *rsp = get_row(y)+(width-2)*num_ch;
        for (int x = 0; x < num_ch; ++x)
        { *ldp++ = *lsp++; *rdp++ = *rsp++; }
      }
    }
    void reset() {
      if (store)
        memset(store, 0, store_size * sizeof(T));
      else
        memset(data, 0, stride * height * sizeof(T));
    }
    void set(T val) {
      T *p = store;
      for (int i = 0; i < store_size; ++i)
        *p++ = val;
    }
//    void set_ext(int val){ this->ext = val;}
    bool exist() { return (store != NULL); }
  private:
    int ext, width, height, stride, num_ch;
    size_t store_size;
    T *store, *data;
  };

  ////////////////////////////////////////////////////////////////////////////
  // A container for a single color component, used with the frame class below
  template <typename T>
  class comp {
  public:
    comp() {
      store_size = ext = width = height = stride = 0;
      store = data = NULL;
    }
    comp(int width, int height, int extension = 16)
    {
      this->store_size = this->ext = this->width = this->height = 0;
      this->stride = 0;
      this->store = this->data = NULL;
      init(width, height, extension);
    }
    ~comp() { if (store) delete[] store; }
    T* get_row(int row) { return data + row * stride; }
    int get_ext() const { return ext; }
    int get_width() const { return width; }
    int get_height() const { return height; }
    int get_stride() const { return stride; }
    void init(int width, int height, int extension = 16)
    {
      this->width = width;
      this->height = height;
      this->ext = extension;
      this->stride = (int)calc_aligned_size<T>(width + 2 * ext);
      size_t size = calc_aligned_size<T>(stride * (height + 2*ext));
      if (store_size < size) {
        if (store) delete[] store;
        store = new T[size];
        store_size = size;
      }
      data = store + ext * stride + ext;
    }
  private:
    int ext, width, height, stride;
    size_t store_size;
    T *store, *data;
  };

  ////////////////////////////////////////////////////////////////////////////
  // A container for a planar frame
  template <typename T>
  class frame {
  public:
    frame() { num_ch = 0; comps = NULL; }
    frame(int width, int height, int num_channels, int extension = 16)
    {
      this->num_ch = 0; this->comps = NULL;
      init(width, height, num_channels, extension);
    }
    ~frame() { if (comps) delete[] comps; }
    void init(int width, int height, int num_ch, int extension = 16) {
      if (comps) delete[] comps;
      this->num_ch = num_ch;
      comps = new comp<T>[num_ch];
      for (int i = 0; i < num_ch; ++i)
        comps[i].init(width, height, extension);
    }
    comp<T>* get_comp(int i) { assert(i<num_ch); return comps + i; }
  private:
    int num_ch;
    comp<T>* comps;
  };

typedef interleaved_frame<flt32> inter_frame_flt32;

}




#endif // SD_FRAME_H
