//
//  map_depth.hpp
//  depth_fun
//
//  Created by Aous Naman on 18/8/17.
//  Copyright © 2017 Aous Naman. All rights reserved.
//

#ifndef SD_MAP_DEPTH_H
#define SD_MAP_DEPTH_H

#include <cmath>
#include "frame.h"

namespace sd {

  struct depth_info  {
    //Stores the depth of a point together with the location of the
    // three points contributing to it, and the amount of that
    // contribution
    depth_info()
    {
      depth = -HUGE_VALF;
      alph = beta = gamm = 0.0f;
      x = y = 0;
      upper_tri = false;
    }
    flt32 depth, alph, beta, gamm;
    si16 x, y;
    bool upper_tri;
  };                      

  //map depth point by point using nearest neighbor
  void map_depth_pbp(const interleaved_frame<flt32>* ref,
                     interleaved_frame<flt32>* tgt,
                     const floc& disp, const int dsubsam);

  void map_depth(const interleaved_frame<flt32>* ref,
                 interleaved_frame<flt32>* tgt,
                 const floc& disp, const int dsubsam, 
                 const float max_expansion);

  void map_image(const interleaved_frame<flt32>* img,
                 const interleaved_frame<flt32>* depth,
                 interleaved_frame<flt32>* tgt,
                 interleaved_frame<flt32>* tmp,
                 const floc& disp, const int dsubsam,
                 const float max_expansion);

  void map_depth_info(const interleaved_frame<flt32>* ref,
                      interleaved_frame<depth_info>* tgt,
                      const floc& disp, const int dsubsam, 
                      const float max_expansion);

  void reverse_map_image(const interleaved_frame<flt32>* tgt_img,
                         interleaved_frame<flt32>* depth,
                         interleaved_frame<flt32>* pred_img,
                         interleaved_frame<depth_info>* tmp,
                         interleaved_frame<flt32>* acc,
                         const floc& disp, const int dsubsam, 
                         const float max_expansion);

  // depth is actually reciprocal depth or normalised disparity.  1/z
  // disp is the disparity between the two views.
  // tmp is a scratch value
  // acc is like a scratch value
  // leave dsubsam as 1. This corresponds to the
  // leave max_expansion as 2.
  typedef interleaved_frame<depth_info> inter_frame_info;
}

#endif //SD_MAP_DEPTH_H
