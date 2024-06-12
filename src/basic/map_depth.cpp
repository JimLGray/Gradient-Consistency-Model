//
//  map_depth.cpp
//  depth_fun
//
//  Created by Aous Naman on 18/8/17.
//  Copyright © 2017 Aous Naman. All rights reserved.
//

#include <cmath>
#include "map_depth.h"

namespace sd {

  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  inline T min(T a, T b) {
    return (a < b ? a : b);
  }

  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  inline T min(T a, T b, T c) {
    T d = a < b ? a : b;
    return (d < c ? d : c);
  }

  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  inline T max(T a, T b) {
    return (a > b ? a : b);
  }

  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  inline T max(T a, T b, T c) {
    T d = a > b ? a : b;
    return (d > c ? d : c);
  }

  //////////////////////////////////////////////////////////////////////////////
  inline float
  find_area(const sd::floc& a, const sd::floc& b, const sd::floc& c)
  {
    float delta_bcy = b.y - c.y;
    float delta_cay = c.y - a.y;
    float delta_aby = a.y - b.y;
    float area = 0.5f * fabsf(a.x*delta_bcy + b.x*delta_cay + c.x*delta_aby);
    return area;
  }

  //////////////////////////////////////////////////////////////////////////////
  void map_depth_pbp(const sd::interleaved_frame<sd::flt32>* ref,
                     sd::interleaved_frame<sd::flt32>* tgt,
                     const sd::floc& odisp, const int dsubsam)
  {
    const sd::floc disp(odisp.x / (float)dsubsam, odisp.y / (float)dsubsam);

    int width = ref->get_width();
    int height = ref->get_height();
    for (int y = 0; y < height; ++y) {
      const float *rp = ref->get_row(y);
      for (int x = 0; x < width; ++x, ++rp) {
        int tx = (int)floor((float)x + disp.x * *rp + 0.5f);
        int ty = (int)floor((float)y + disp.y * *rp + 0.5f);
        if (tx >= 0 && tx < width && ty >= 0 && ty < height)
        {
          float *tp = tgt->get_row(ty) + tx;
          *tp = sd::max(*rp, *tp);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  void map_depth(const sd::interleaved_frame<sd::flt32>* ref,
                 sd::interleaved_frame<sd::flt32>* tgt,
                 const sd::floc& odisp, const int dsubsam, 
                 const float max_expansion)
  {
//     for (int y = 0; y < ref->get_height(); ++y)
//     {
//         const float *sp = ref->get_row(y);
//         float *dp = tgt->get_row(y);
//         for (int x = 0; x < ref->get_width(); ++x)
//             *dp++ = *sp++;
//     }
//     return;
    
    float twice_max_expansion = 2.0f * max_expansion;
    twice_max_expansion /= (float)dsubsam * (float)dsubsam;
    const sd::floc disp(odisp.x / (float)dsubsam, odisp.y / (float)dsubsam);

    int width = ref->get_width();
    int height = ref->get_height();
    int ext = ref->get_ext();
    for (int y = 0; y < height; ++y) {
      const float *rp0 = ref->get_row(y);
      const float *rp1 = ref->get_row(y + 1);
      for (int x = 0; x < width; ++x, ++rp0, ++rp1) {

        // quad point mapping
        sd::floc p1 = sd::floc((float)x+disp.x*rp0[0],   (float)y+disp.y*rp0[0]);
        sd::floc p2 = sd::floc((float)x+disp.x*rp0[1]+1, (float)y+disp.y*rp0[1]);
        sd::floc p3 = sd::floc((float)x+disp.x*rp1[0],   (float)y+disp.y*rp1[0]+1);

        //upper left triangle
        float delta_13x = p1.x - p3.x;
        float delta_23x = p2.x - p3.x;
        float delta_13y = p1.y - p3.y;
        float delta_23y = p2.y - p3.y;
        float twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        float inv_twice_area = 1.0f / twice_area;
        // no expansion or contraction when twice area is 1
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));

          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph >= 0 && beta >= 0 && gamm >= 0)
                {
                  float val = alph * rp0[0] + beta * rp0[1] + gamm * rp1[0];
                  float *tp = tgt->get_row(p.y) + p.x;
                  val *= inv_twice_area;
                  tp[0] = max(tp[0], val);
                }
              }
            }
          }
        }

        //bottom right triangle
        p1 = sd::floc((float)x + disp.x * rp1[1] + 1, (float)y + disp.y * rp1[1] + 1);
        delta_13x = p1.x - p3.x;
        delta_23x = p2.x - p3.x;
        delta_13y = p1.y - p3.y;
        delta_23y = p2.y - p3.y;
        twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        inv_twice_area = 1.0f / twice_area;
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));

          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph <= 0 && beta <= 0 && gamm <= 0)
                {
                  float val = alph * rp1[1] + beta * rp0[1] + gamm * rp1[0];
                  float *tp = tgt->get_row(p.y) + p.x;
                  val *= inv_twice_area;
                  tp[0] = max(tp[0], val);
                }
              }
            }
          }
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  void map_image(const sd::interleaved_frame<sd::flt32>* img,
                 const sd::interleaved_frame<sd::flt32>* depth,
                 sd::interleaved_frame<sd::flt32>* tgt,
                 sd::interleaved_frame<sd::flt32>* tmp,
                 const sd::floc& odisp, const int dsubsam, 
                 const float max_expansion)
  {
    assert(tgt->get_num_ch() == img->get_num_ch());
    assert(tgt->get_width() == img->get_width());
    assert(tgt->get_height() == img->get_height());

    tmp->init(depth->get_width(), depth->get_height(), 1, 16);
    tmp->set(-HUGE_VALF);
    map_depth(depth, tmp, odisp, dsubsam, max_expansion);

    float twice_max_expansion = 2.0f * max_expansion;
    twice_max_expansion /= (float)dsubsam * (float)dsubsam;
    const sd::floc disp(odisp.x / (float)dsubsam, odisp.y / (float)dsubsam);

    int nc = img->get_num_ch();
    int width = depth->get_width();
    int height = depth->get_height();
    int ext = depth->get_ext();
    for (int y = 0; y < height; ++y) {
      const float *dp0 = depth->get_row(y);
      const float *dp1 = depth->get_row(y + 1);
      const float *ip0 = img->get_row(y);
      const float *ip1 = img->get_row(y + 1);
      for (int x = 0; x < width; ++x, ++dp0, ++dp1, ip0+=nc, ip1+=nc)
      {
        // quad point mapping
        sd::floc p1 = sd::floc((float)x+disp.x*dp0[0],   (float)y+disp.y*dp0[0]);
        sd::floc p2 = sd::floc((float)x+disp.x*dp0[1]+1, (float)y+disp.y*dp0[1]);
        sd::floc p3 = sd::floc((float)x+disp.x*dp1[0],   (float)y+disp.y*dp1[0]+1);

        //upper left triangle
        float delta_13x = p1.x - p3.x;
        float delta_23x = p2.x - p3.x;
        float delta_13y = p1.y - p3.y;
        float delta_23y = p2.y - p3.y;
        float twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        float inv_twice_area = 1.0f / twice_area;
        // no expansion or contraction when twice area is 1
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));
          
          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph >= 0 && beta >= 0 && gamm >= 0)
                {
                  float d = alph * dp0[0] + beta * dp0[1] + gamm * dp1[0];
                  float *c = tmp->get_row(p.y) + p.x;
                  if (d * inv_twice_area == *c) {
                    float val, *tp = tgt->get_row(p.y) + p.x * nc;
                    for (int i = 0; i < nc; ++i) {
                      val = alph * ip0[i] + beta * ip0[i+nc] + gamm * ip1[i];
                      tp[i] = val * inv_twice_area;
                    }
                  }
                }
              }
            }
          }
        }

        //bottom right triangle
        p1 = sd::floc((float)x + disp.x * dp1[1] + 1, (float)y + disp.y * dp1[1] + 1);
        delta_13x = p1.x - p3.x;
        delta_23x = p2.x - p3.x;
        delta_13y = p1.y - p3.y;
        delta_23y = p2.y - p3.y;
        twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        inv_twice_area = 1.0f / twice_area;
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));

          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph <= 0 && beta <= 0 && gamm <= 0)
                {
                  float d = alph * dp1[1] + beta * dp0[1] + gamm * dp1[0];
                  float *c = tmp->get_row(p.y) + p.x;
                  if (d * inv_twice_area == *c) {
                    float val, *tp = tgt->get_row(p.y) + p.x * nc;
                    for (int i = 0; i < nc; ++i) {
                      val = alph * ip1[i+nc] + beta * ip0[i+nc] + gamm * ip1[i];
                      tp[i] = val * inv_twice_area;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  void map_depth_info(const interleaved_frame<flt32>* ref,
                      interleaved_frame<depth_info>* tgt,
                      const floc& odisp, const int dsubsam, 
                      const float max_expansion)
  {
    float twice_max_expansion = 2.0f * max_expansion;
    twice_max_expansion /= (float)dsubsam * (float)dsubsam;
    const sd::floc disp(odisp.x / (float)dsubsam, odisp.y / (float)dsubsam);

    int width = ref->get_width();
    int height = ref->get_height();
    int ext = ref->get_ext();
    for (int y = 0; y < height; ++y) {
      const float *rp0 = ref->get_row(y);
      const float *rp1 = ref->get_row(y + 1);
      for (int x = 0; x < width; ++x, ++rp0, ++rp1) {

        // quad point mapping
        sd::floc p1 = sd::floc(x+disp.x*rp0[0],   y+disp.y*rp0[0]);
        sd::floc p2 = sd::floc(x+disp.x*rp0[1]+1, y+disp.y*rp0[1]);
        sd::floc p3 = sd::floc(x+disp.x*rp1[0],   y+disp.y*rp1[0]+1);

        //upper left triangle
        float delta_13x = p1.x - p3.x;
        float delta_23x = p2.x - p3.x;
        float delta_13y = p1.y - p3.y;
        float delta_23y = p2.y - p3.y;
        float twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        float inv_twice_area = 1.0f / twice_area;
        // no expansion or contraction when twice area is 1
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));

          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph >= 0 && beta >= 0 && gamm >= 0)
                {
                  float val = alph * rp0[0] + beta * rp0[1] + gamm * rp1[0];
                  val *= inv_twice_area;
                  depth_info *tp = tgt->get_row(p.y) + p.x;
                  if (val > tp->depth)
                  {
                    tp->depth = val;
                    tp->alph = alph * inv_twice_area;
                    tp->beta = beta * inv_twice_area;
                    tp->gamm = gamm * inv_twice_area;
                    tp->x = (si16)x;
                    tp->y = (si16)y;
                    tp->upper_tri = true;
                  }
                }
              }
            }
          }
        }

        //bottom right triangle
        p1 = sd::floc(x + disp.x * rp1[1] + 1, y + disp.y * rp1[1] + 1);
        delta_13x = p1.x - p3.x;
        delta_23x = p2.x - p3.x;
        delta_13y = p1.y - p3.y;
        delta_23y = p2.y - p3.y;
        twice_area = delta_23y * delta_13x - delta_23x * delta_13y;
        inv_twice_area = 1.0f / twice_area;
        if (twice_area != 0 && fabs(twice_area) < twice_max_expansion)
        {
          int sx = (int)floor(min(p1.x, p2.x, p3.x));
          int sy = (int)floor(min(p1.y, p2.y, p3.y));
          int ex = (int)ceil(max(p1.x, p2.x, p3.x));
          int ey = (int)ceil(max(p1.y, p2.y, p3.y));

          sx = max(sx, -ext);
          ex = min(ex, width + ext);
          sy = max(sy, -ext);
          ey = min(ey, height + ext);

          //if (sx > -ext && sy > -ext && ex < width + ext && ey < height + ext)
          {
            sd::loc p;
            for (p.y = sy; p.y < ey; ++p.y)
            {
              float delta_03y = (float)p.y - p3.y;
              for (p.x = sx; p.x < ex; ++p.x)
              {
                float delta_03x = (float)p.x - p3.x;

                float alph = delta_23y * delta_03x - delta_23x * delta_03y;
                float beta = delta_13x * delta_03y - delta_13y * delta_03x;
                float gamm = twice_area - alph - beta;
                //either all alph, beta, gamm negative or positive
                //zero means that the point is on the line, which is permissible
                // if (alph * beta >= 0 && alph * gamm >= 0 && beta * gamm >= 0)
                // if ((alph >= 0 && beta >= 0 && gamm >= 0) ||
                //     (alph <= 0 && beta <= 0 && gamm <= 0))
                if (alph <= 0 && beta <= 0 && gamm <= 0)
                {
                  float val = alph * rp1[1] + beta * rp0[1] + gamm * rp1[0];
                  val *= inv_twice_area;
                  depth_info *tp = tgt->get_row(p.y) + p.x;
                  if (val > tp->depth) {
                    tp->depth = val;
                    tp->alph = alph * inv_twice_area;
                    tp->beta = beta * inv_twice_area;
                    tp->gamm = gamm * inv_twice_area;
                    tp->x = (si16)x;
                    tp->y = (si16)y;
                    tp->upper_tri = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  void reverse_map_image(const interleaved_frame<flt32>* tgt_img,
                         interleaved_frame<flt32>* depth,
                         interleaved_frame<flt32>* pred_img,
                         interleaved_frame<depth_info>* tmp,
                         interleaved_frame<flt32>* acc,
                         const floc& odisp, const int dsubsam, 
                         const float max_expansion)
  {
    assert(tgt_img->get_num_ch() == pred_img->get_num_ch());
    assert(tgt_img->get_width() == pred_img->get_width());
    assert(tgt_img->get_height() == pred_img->get_height());

    tmp->init(depth->get_width(), depth->get_height(), 1, 16);
    tmp->set(depth_info());
    depth->zohold_extend();
    map_depth_info(depth, tmp, odisp, dsubsam, max_expansion);

    acc->init(depth->get_width(), depth->get_height(), 1, 16);
    acc->reset();
    pred_img->reset();

    int width = tgt_img->get_width();
    int height = tgt_img->get_height();
    int num_ch = tgt_img->get_num_ch();
    int sstr = tgt_img->get_stride();
    int astr = acc->get_stride();
    for (int y = 0; y < height; ++y) {
      depth_info *d = tmp->get_row(y);
      const float *sp = tgt_img->get_row(y);
      for (int x = 0; x < width; ++x, ++d, sp+=num_ch) {
        if (std::isinf(d->depth) == false)
        {
          //alpha
          float *dp = pred_img->get_row(d->y) + d->x * num_ch;
          float *ap = acc->get_row(d->y) + d->x;
          int al = d->upper_tri ? 0 : astr + 1;
          ap[al] += d->alph;
          int dl = d->upper_tri ? 0 : sstr + num_ch;
          for (int c = 0; c < num_ch; ++c)
            dp[dl + c] += d->alph * sp[c];

          //beta
          ap[1] += d->beta;
          for (int c = 0; c < num_ch; ++c)
            dp[num_ch + c] += d->beta * sp[c];

          //gamma
          ap[astr] += d->gamm;
          for (int c = 0; c < num_ch; ++c)
            dp[sstr + c] += d->gamm * sp[c];
        }
      }
    }

    for (int y = 0; y < height - 1; ++y) {
      float *dp = pred_img->get_row(y);
      float *ap = acc->get_row(y);
      for (int x = 0; x < width - 1; ++x, ++ap, dp+=num_ch) {
        if (*ap != 0.0f)
          for (int c = 0; c < num_ch; ++c)
            dp[c] /= *ap;
        else
          for (int c = 0; c < num_ch; ++c)
            dp[c] = -HUGE_VALF;
      }
    }
  }


}

