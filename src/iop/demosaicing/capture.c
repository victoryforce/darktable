/*
    This file is part of darktable,
    Copyright (C) 2025 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/* remarks:
    Credits go to: Ingo Weyrich (heckflosse67@gmx.de), he implemented the original algorithm for rawtherapee.

   1) - The gaussian convolution filters take the coeffs from precalculated data in gd->gauss_coeffs,
        we have CHAR_MAX kernels with a sigma step of CAPTURE_GAUSS_FRACTION.
      - The chosen kernel is selected per pixel via an index map, this is derived from cs_radius, cs_boost
        and distance from image centre.
      - using the index map improves performance and allows runtime modification of the used per pixel
        gaussian sigma.
      - Note: this is similar to the per-tile sigma in the RT implementation.
   2) It's currently not planned to increase the maximum sigma so we can stay with the 9x9 kernels.
   3) Reminders and possibly left to do:
      - halo suppression at very strong gradients?
      - automatic noise detection or reduction?
      - can we auto-stop? per pixel?
      - Internal CPU code tiling for performance? tile size would be the same as for rcd
*/

#ifdef __GNUC__
  #pragma GCC push_options
  #pragma GCC optimize ("fp-contract=fast", "finite-math-only", "no-math-errno")
#endif

#define CAPTURE_KERNEL_ALIGN 32
#define CAPTURE_GAUSS_FRACTION 0.01f
#define CAPTURE_YMIN 0.001f
#define CAPTURE_CFACLIP 0.9f

static inline void _calc_9x9_gauss_coeffs(float *coeffs, const float sigma)
{
  float kernel[9][9];
  const float range = 4.5f * 4.5f;
  const float temp = -2.0f * sigma * sigma;
  float sum = 0.0;
  for(int k = -4; k < 5; k++)
  {
    for(int j = -4; j < 5; j++)
    {
      const float rad = (float)(k*k + j*j);
      if(rad <= range)
      {
        kernel[k + 4][j + 4] = expf(rad / temp);
        sum += kernel[k + 4][j + 4];
      }
      else
        kernel[k + 4][j + 4] = 0.0f;
    }
  }

  for(int k = 0; k < 5; k++)
    for(int j = 0; j < 5; j++)
      coeffs[5*k+j] = kernel[k+4][j+4] / sum;
}

static inline unsigned char _sigma_to_index(const float sigma)
{
  return CLAMP((int)(sigma / CAPTURE_GAUSS_FRACTION), 0, UCHAR_MAX);
}

// provide an index map so the convolution kernels can easily get the correct coeffs
static unsigned char *_cs_precalc_gauss_idx(dt_iop_module_t *self,
                                            const dt_iop_roi_t *const roi,
                                            const float isigma,
                                            const float boost,
                                            const float centre)
{
  const dt_image_t *img = &self->dev->image_storage;
  const int rwidth = img->p_width / 2;
  const int rheight = img->p_height / 2;
  const float mdim = MIN(rwidth, rheight);
  const int width = roi->width;
  const int height = roi->height;
  const int dy = roi->y;
  const int dx = roi->x;
  unsigned char *table = dt_alloc_aligned((size_t)height * width);
  if(!table) return NULL;

  const float cboost = 1.0f + 8.0f * sqrf(centre);
  DT_OMP_FOR()
  for(int row = 0; row < height; row++)
  {
    const float frow = row + dy - rheight;
    for(int col = 0; col < width; col++)
    {
      const float fcol = col + dx - rwidth;
      const float sc = sqrtf(frow * frow + fcol * fcol) / mdim;
      const float corr = cboost * boost * sqrf(MAX(0.0f, sc - 0.5f - centre));

      // also special care for the image borders
      const float sigma = (isigma + corr) * 0.125f * (float)MIN(8, MIN(height-row-1, MIN(width-col-1, MIN(col, row))));
      table[row * width + col] = _sigma_to_index(sigma);
    }
  }
  return table;
}

#define RAWEPS 0.005f
static float _calcRadiusBayer(const float *in,
                              const int width,
                              const int height,
                              const float lowerLimit,
                              const float upperLimit,
                              const uint32_t filters)
{
  const unsigned int fc[2] = {FC(0, 0, filters), FC(1, 0, filters)};
  float maxRatio = 1.0f;
  DT_OMP_FOR(reduction(max: maxRatio))
  for(int row = 4; row < height - 4; ++row)
  {
    for(int col = 5 + (fc[row & 1] & 1); col < width - 4; col += 2)
    {
      const float *cfa = in + row*width + col;
      const float val00 = cfa[0];
      if(val00 > RAWEPS)
      {
        const float val1m1 = cfa[width-1];
        const float val1p1 = cfa[width+1];
        const float maxVal0 = MAX(val00, val1m1);
        if(val1m1 > RAWEPS && maxVal0 > lowerLimit)
        {
          const float minVal = MIN(val00, val1m1);
          if(maxVal0 > maxRatio * minVal)
          {
            gboolean clipped = FALSE;
            if(maxVal0 == val00)
            { // check for influence by clipped green in neighborhood
              if(MAX(MAX(cfa[-width-1], cfa[-width+1]), val1p1) >= upperLimit)
                clipped = TRUE;
            }
            else
            { // check for influence by clipped green in neighborhood
              if(MAX(MAX(MAX(cfa[-2], val00), cfa[2*width-2]), cfa[2*width]) >= upperLimit)
                clipped = TRUE;
            }
            if(!clipped)
              maxRatio = maxVal0 / minVal;
          }
        }

        const float maxVal1 = MAX(val00, val1p1);
        if(val1p1 > RAWEPS && maxVal1 > lowerLimit)
        {
          const float minVal = MIN(val00, val1p1);
          if(maxVal1 > maxRatio * minVal)
          {
            if(maxVal1 == val00)
            { // check for influence by clipped green in neighborhood
              if(MAX(MAX(cfa[-width-1], cfa[-width+1]), val1p1) >= upperLimit)
                continue;
             }
            else
            { // check for influence by clipped green in neighborhood
              if(MAX(MAX(MAX(val00, cfa[2]), cfa[2*width]), cfa[2*width+2]) >= upperLimit)
                continue;
             }
            maxRatio = maxVal1 / minVal;
          }
        }
      }
    }
  }
  return sqrtf(1.0f / logf(maxRatio));
}

static float _calcRadiusXtrans(const float *in,
                               const float lowerLimit,
                               const float upperLimit,
                               const dt_iop_roi_t *const roi,
                               const uint8_t(*const xtrans)[6])
{
  const int width = roi->width;
  const int height = roi->height;

  int startx, starty;
  gboolean found = FALSE;
  for(starty = 6; starty < 12 && !found; starty++)
  {
    for(startx = 6; startx < 12 && !found; startx++)
    {
      if(FCxtrans(starty, startx, roi, xtrans) == 1)
      {
        if(FCxtrans(starty, startx - 1, roi, xtrans) != FCxtrans(starty, startx + 1, roi, xtrans))
        {
          if(FCxtrans(starty -1, startx, roi, xtrans) != 1)
          {
            if(FCxtrans(starty, startx -1, roi, xtrans) != 1)
            {
              found = TRUE;
              break;
            }
          }
        }
      }
    }
  }

  float maxRatio = 1.0f;
  DT_OMP_FOR(reduction(max: maxRatio))
  for(int row = starty + 2; row < height - 4; row += 3)
  {
    for(int col = startx + 2; col < width - 4; col += 3)
    {
      const float *cfa = in + row*width + col;
      const float valp1p1 = cfa[width+1];
      const gboolean squareClipped = MAX(MAX(MAX(valp1p1, cfa[width+2]), cfa[2*width+1]), cfa[2*width+2]) >= upperLimit;
      const float greenSolitary = cfa[0];
      if(greenSolitary > RAWEPS && MAX(cfa[-width-1], cfa[-width+1]) < upperLimit)
      {
        if(greenSolitary < upperLimit)
        {
          const float valp1m1 = cfa[width-1];
          if(valp1m1 > RAWEPS && MAX(MAX(MAX(cfa[width-2], valp1m1), cfa[2*width-2]), cfa[width-1]) < upperLimit)
          {
            const float maxVal = MAX(greenSolitary, valp1m1);
            if(maxVal > lowerLimit)
            {
              const float minVal = MIN(greenSolitary, valp1m1);
              if(maxVal > maxRatio * minVal)
                maxRatio = maxVal / minVal;
            }
          }
          if(valp1p1 > RAWEPS && !squareClipped)
          {
            const float maxVal = MAX(greenSolitary, valp1p1);
            if(maxVal > lowerLimit)
            {
              const float minVal = MIN(greenSolitary, valp1p1);
              if(maxVal > maxRatio * minVal)
                maxRatio = maxVal / minVal;
            }
          }
        }
      }

      if(!squareClipped)
      {
        const float valp2p2 = cfa[2*width+2];
        if(valp2p2 > RAWEPS)
        {
          if(valp1p1 > RAWEPS)
          {
            const float maxVal = MAX(valp1p1, valp2p2);
            if(maxVal > lowerLimit)
            {
              const float minVal = MIN(valp1p1, valp2p2);
              if(maxVal > maxRatio * minVal)
                 maxRatio = maxVal / minVal;
            }
          }
          const float greenSolitaryRight = cfa[3*width+3];
          if(MAX(MAX(greenSolitaryRight, cfa[4*width+2]), cfa[4*width+4]) < upperLimit)
          {
            if(greenSolitaryRight > RAWEPS)
            {
              const float maxVal = MAX(greenSolitaryRight, valp2p2);
              if(maxVal > lowerLimit)
              {
                const float minVal = MIN(greenSolitaryRight, valp2p2);
                if(maxVal > maxRatio * minVal)
                  maxRatio = maxVal / minVal;
              }
            }
          }
        }
        const float valp1p2 = cfa[width+2];
        const float valp2p1 = cfa[2*width+1];
        if(valp2p1 > RAWEPS)
        {
          if(valp1p2 > RAWEPS)
          {
            const float maxVal = MAX(valp1p2, valp2p1);
            if(maxVal > lowerLimit)
            {
              const float minVal = MIN(valp1p2, valp2p1);
              if(maxVal > maxRatio * minVal)
                maxRatio = maxVal / minVal;
            }
          }
          const float greenSolitaryLeft = cfa[3*width];
          if(MAX(MAX(greenSolitaryLeft, cfa[4*width-1]), cfa[4*width+1]) < upperLimit)
          {
            if(greenSolitaryLeft > RAWEPS)
            {
              const float maxVal = MAX(greenSolitaryLeft, valp2p1);
              if(maxVal > lowerLimit)
              {
                const float minVal = MIN(greenSolitaryLeft, valp2p1);
                if(maxVal > maxRatio * minVal)
                  maxRatio = maxVal / minVal;
              }
            }
          }
        }
      }
    }
  }
  return sqrtf(1.0f / logf(maxRatio));
}
#undef RAWEPS

DT_OMP_DECLARE_SIMD(aligned(in, out, blend, kernels:64))
static inline void _blur_mul(const float *const in,
                             float *out,
                             const float *blend,
                             const float *const kernels,
                             const unsigned char *const table,
                             const int w1,
                             const int height)
{
  const int w2 = 2 * w1;
  const int w3 = 3 * w1;
  const int w4 = 4 * w1;

  DT_OMP_FOR()
  for(int row = 0; row < height; row++)
  {
    for(int col = 0; col < w1; col++)
    {
      const size_t i = (size_t)row * w1 + col;
      if(blend[i] > 0.0f)
      {
        const float *kern = kernels + CAPTURE_KERNEL_ALIGN * table[i];
        float val = 0.0f;
        if(col >= 4 && row >= 4 && col < w1 - 4 && row < height - 4)
        {
          const float *d = in + i;
          val =
              kern[10+4] * (d[-w4-2] + d[-w4+2] + d[-w2-4] + d[-w2+4] + d[w2-4] + d[w2+4] + d[w4-2] + d[w4+2]) +
              kern[5 +4] * (d[-w4-1] + d[-w4+1] + d[-w1-4] + d[-w1+4] + d[w1-4] + d[w1+4] + d[w4-1] + d[w4+1]) +
              kern[4]    * (d[-w4  ] + d[   -4] + d[    4] + d[ w4  ]) +
              kern[15+3] * (d[-w3-3] + d[-w3+3] + d[ w3-3] + d[ w3+3]) +
              kern[10+3] * (d[-w3-2] + d[-w3+2] + d[-w2-3] + d[-w2+3] + d[w2-3] + d[w2+3] + d[w3-2] + d[w3+2]) +
              kern[ 5+3] * (d[-w3-1] + d[-w3+1] + d[-w1-3] + d[-w1+3] + d[w1-3] + d[w1+3] + d[w3-1] + d[w3+1]) +
              kern[   3] * (d[-w3  ] + d[   -3] + d[    3] + d[ w3  ]) +
              kern[10+2] * (d[-w2-2] + d[-w2+2] + d[ w2-2] + d[ w2+2]) +
              kern[ 5+2] * (d[-w2-1] + d[-w2+1] + d[-w1-2] + d[-w1+2] + d[w1-2] + d[w1+2] + d[w2-1] + d[w2+1]) +
              kern[   2] * (d[-w2  ] + d[   -2] + d[    2] + d[ w2  ]) +
              kern[ 5+1] * (d[-w1-1] + d[-w1+1] + d[ w1-1] + d[ w1+1]) +
              kern[   1] * (d[-w1  ] + d[   -1] + d[    1] + d[ w1  ]) +
              kern[   0] * (d[0]);
        }
        else
        {
          for(int ir = -4; ir <= 4; ir++)
          {
            const int irow = row+ir;
            if(irow >= 0 && irow < height)
            {
              for(int ic = -4; ic <= 4; ic++)
              {
                const int icol = col+ic;
                if(icol >=0 && icol < w1)
                  val += kern[5 * ABS(ir) + ABS(ic)] * in[(size_t)irow * w1 + icol];
              }
            }
          }
        }
        out[i] *= val;
      }
      // if blend value is too low we don't have to copy data as we also didn't in _blur_div
      // and we just keep the original
    }
  }
}

DT_OMP_DECLARE_SIMD(aligned(in, out, luminance, blend, kernels :64))
static inline void _blur_div(const float *const in,
                             float *out,
                             const float *const luminance,
                             const float *blend,
                             const float *const kernels,
                             const unsigned char *const table,
                             const int w1,
                             const int height)
{
  const int w2 = 2 * w1;
  const int w3 = 3 * w1;
  const int w4 = 4 * w1;

  DT_OMP_FOR()
  for(int row = 0; row < height; row++)
  {
    for(int col = 0; col < w1; col++)
    {
      const size_t i = (size_t)row * w1 + col;
      if(blend[i] > 0.0f)
      {
        const float *kern = kernels + CAPTURE_KERNEL_ALIGN * table[i];
        float val = 0.0f;
        if(col >= 4 && row >= 4 && col < w1 - 4 && row < height - 4)
        {
          const float *d = in + i;
          val =
              kern[10+4] * (d[-w4-2] + d[-w4+2] + d[-w2-4] + d[-w2+4] + d[w2-4] + d[w2+4] + d[w4-2] + d[w4+2]) +
              kern[5 +4] * (d[-w4-1] + d[-w4+1] + d[-w1-4] + d[-w1+4] + d[w1-4] + d[w1+4] + d[w4-1] + d[w4+1]) +
              kern[4]    * (d[-w4  ] + d[   -4] + d[    4] + d[ w4  ]) +
              kern[15+3] * (d[-w3-3] + d[-w3+3] + d[ w3-3] + d[ w3+3]) +
              kern[10+3] * (d[-w3-2] + d[-w3+2] + d[-w2-3] + d[-w2+3] + d[w2-3] + d[w2+3] + d[w3-2] + d[w3+2]) +
              kern[ 5+3] * (d[-w3-1] + d[-w3+1] + d[-w1-3] + d[-w1+3] + d[w1-3] + d[w1+3] + d[w3-1] + d[w3+1]) +
              kern[   3] * (d[-w3  ] + d[   -3] + d[    3] + d[ w3  ]) +
              kern[10+2] * (d[-w2-2] + d[-w2+2] + d[ w2-2] + d[ w2+2]) +
              kern[ 5+2] * (d[-w2-1] + d[-w2+1] + d[-w1-2] + d[-w1+2] + d[w1-2] + d[w1+2] + d[w2-1] + d[w2+1]) +
              kern[   2] * (d[-w2  ] + d[   -2] + d[    2] + d[ w2  ]) +
              kern[ 5+1] * (d[-w1-1] + d[-w1+1] + d[ w1-1] + d[ w1+1]) +
              kern[   1] * (d[-w1  ] + d[   -1] + d[    1] + d[ w1  ]) +
              kern[   0] * (d[0]);
        }
        else
        {
          for(int ir = -4; ir <= 4; ir++)
          {
            const int irow = row+ir;
            if(irow >= 0 && irow < height)
            {
              for(int ic = -4; ic <= 4; ic++)
              {
                const int icol = col+ic;
                if(icol >=0 && icol < w1)
                  val += kern[5 * ABS(ir) + ABS(ic)] * in[(size_t)irow * w1 + icol];
              }
            }
          }
        }
        out[i] = luminance[i] / MAX(val, CAPTURE_YMIN);
      }
    }
  }
}

static void _prepare_blend(const float *cfa,
                           const float *rgb,
                           const uint32_t filters,
                           const uint8_t (*const xtrans)[6],
                           const dt_iop_roi_t *const roi,
                           float *mask,
                           float *Yold,
                           const float *whites,
                           const int w1,
                           const int height)
{
  dt_iop_image_fill(mask, 1.0f, w1, height, 1);
  const int w2 = 2 * w1;
  // Photometric/digital ITU BT.709
  const dt_aligned_pixel_t flum = { 0.212671f, 0.715160f, 0.072169f, 0.0f };
  DT_OMP_FOR(collapse(2))
  for(size_t row = 0; row < height; row++)
  {
    for(size_t col = 0; col < w1; col++)
    {
      const size_t k = row * w1 + col;
      dt_aligned_pixel_t yw;
      for_each_channel(c) yw[c] = flum[c] * rgb[k*4+c];
      Yold[k] = MAX(0.0f, yw[0] + yw[1] + yw[2]);
      if(row > 1 && col > 1 && row < height-2 && col < w1-2)
      {
        const int color = (filters == 9u) ? FCxtrans(row, col, roi, xtrans) : FC(row, col, filters);
        if(cfa[k] > whites[color] || Yold[k] < CAPTURE_YMIN)
        {
          mask[k-w2-1] = mask[k-w2]   = mask[k-w2+1] =
          mask[k-w1-2] = mask[k-w1-1] = mask[k-w1]   = mask[k-w1+1] = mask[k-w1+2] =
          mask[k-2]    = mask[k-1]    = mask[k]      = mask[k+1]    = mask[k+2] =
          mask[k+w1-2] = mask[k+w1-1] = mask[k+w1]   = mask[k+w1+1] = mask[k+w1+2] =
          mask[k+w2-1] = mask[k+w2]   = mask[k+w2+1] = 0.0f;
        }
      }
      else
        mask[k] = 0.0f;
    }
  }
}

static void _modify_blend(float *blend,
                          float *Yold,
                          float *luminance,
                          const float dthresh,
                          const int width,
                          const int height)
{
  const float threshold = 0.6f * sqrf(dthresh);
  const float tscale = 200.0f;
  const float offset = -2.5f + tscale * threshold / 2.0f;
  DT_OMP_FOR()
  for(int irow = 0; irow < height; irow++)
  {
    const int row = CLAMP(irow, 2, height-3);
    for(int icol = 0; icol < width; icol++)
    {
      const int col = CLAMP(icol, 2, width-3);
      const size_t k = (size_t)irow * width + icol;
      float sum = 0.0f;
      float sum_sq = 0.0f;
      for(int y = row-1; y < row+2; y++)
      {
        for(int x = col-2; x < col+3; x++)
        {
          sum += Yold[(size_t)y*width + x];
          sum_sq += sqrf(Yold[(size_t)y*width + x]);
        }
      }
      for(int x = col-1; x < col+2; x++)
      {
        sum += Yold[(size_t)(row-2)*width + x];
        sum_sq += sqrf(Yold[(size_t)(row-2)*width + x]);
        sum += Yold[(size_t)(row+2)*width + x];
        sum_sq += sqrf(Yold[(size_t)(row+2)*width + x]);
      }
      // we don't have to count locations as it's always 21
      const float sum_of_squares = MAX(0.0f, sum_sq - sqrf(sum) / 21.0f);
      const float std_deviation = sqrtf(sum_of_squares / 21.0f);
      const float mean = MAX(NORM_MIN, sum / 21.0f);
      const float modified_coef_variation = std_deviation / sqrtf(mean);
      const float t = logf(1.0f + modified_coef_variation);
      const float weight = 1.0f / (1.0f + expf(offset - tscale * t));
      blend[k] = CLIP(blend[k] * 1.01011f * (weight - 0.01f));
      luminance[k] = Yold[k];
    }
  }
}

void _capture_sharpen(dt_iop_module_t *self,
                      dt_dev_pixelpipe_iop_t *const piece,
                      float *const in,
                      float *out,
                      const dt_iop_roi_t *const roi,
                      const gboolean show_variance_mask,
                      const gboolean show_sigma_mask)
{
  dt_dev_pixelpipe_t *pipe = piece->pipe;

  const size_t width = roi->width;
  const size_t height = roi->height;
  const size_t pixels = width * height;
  const dt_iop_demosaic_data_t *d = piece->data;
  const dt_iop_demosaic_global_data_t *gd = self->global_data;
  dt_iop_demosaic_gui_data_t *g = self->gui_data;

  if(pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL)
  {
    const gboolean hqthumb = _get_thumb_quality(pipe->final_width, pipe->final_height);
    if(!hqthumb) return;
  }

  if(d->cs_iter < 1 && !show_variance_mask && !show_sigma_mask) return;

  const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])pipe->dsc.xtrans;
  const uint32_t filters = pipe->dsc.filters;
  const dt_iop_buffer_dsc_t *dsc = &pipe->dsc;
  const gboolean wbon = dsc->temperature.enabled;
  const dt_aligned_pixel_t icoeffs = { wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[0] : CAPTURE_CFACLIP,
                                       wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[1] : CAPTURE_CFACLIP,
                                       wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[2] : CAPTURE_CFACLIP,
                                       0.0f };
  const gboolean fullpipe = pipe->type & DT_DEV_PIXELPIPE_FULL;
  const gboolean autoradius = fullpipe && g && g->autoradius;
  const float old_radius = d->cs_radius;
  float radius = old_radius;
  if(autoradius || radius < 0.01f)
  {
    radius = filters != 9u
              ? _calcRadiusBayer(in, width, height, 0.01f, 1.0f, filters)
              : _calcRadiusXtrans(in, 0.01f, 1.0f, roi, xtrans);
    const gboolean valid = radius > 0.1f && radius < 1.0f;

    dt_print_pipe(DT_DEBUG_PIPE, filters != 9u ? "bayer autoradius" : "xtrans autoradius",
      pipe, self, DT_DEVICE_CPU, roi, NULL, "autoradius=%.2f", radius);

    if(!feqf(radius, old_radius, 0.005f) && valid)
    {
      if(fullpipe)
      {
        if(g)
        {
          dt_control_log(_("calculated capture radius"));
          g->autoradius = TRUE;
        }
        dt_iop_demosaic_params_t *p = self->params;
        p->cs_radius = radius;
      }
    }
    else if(g) g->autoradius = FALSE;
  }

  unsigned char *gauss_idx = NULL;
  gboolean error = TRUE;

  float *luminance = dt_alloc_align_float(pixels);
  float *tmp2 = dt_alloc_align_float(pixels);
  float *tmp1 = dt_alloc_align_float(pixels);
  float *blendmask = dt_alloc_align_float(pixels);
  if(!luminance || !tmp2 || !tmp1 || !blendmask)
    goto finalize;

  // tmp2 will hold the temporary clipmask, tmp1 holds Y data
  _prepare_blend(in, out, filters, xtrans, roi, tmp2, tmp1, icoeffs, width, height);
  // modify clipmask in tmp2 according to Y variance, also write L to luminance
  _modify_blend(tmp2, tmp1, luminance, d->cs_thrs, width, height);

  dt_gaussian_fast_blur(tmp2, blendmask, width, height, 2.0f, 0.0f, 1.0f, 1);

  // after the blur, very tiny edges will not get enough strength of sharpening
  // use the maximum of (unblurred,blurred) values.
  DT_OMP_FOR()
  for(size_t k = 0; k < pixels; k++)
  {
    // difference between the calculated blend from modified_blend, and the blurred value
    // if the difference is large, the local value was reduced too much as a result of the blurring
    // use a weighted mean of the unblurred (aka tmp2) and the blurred (aka blendmask)
    const float diff = tmp2[k] - blendmask[k];
    const float w_tmp2 = 1.0f / (1.0f + expf(5.0f - 10.0f * diff));
    blendmask[k] = CLIP(w_tmp2 * tmp2[k] + (1.0f - w_tmp2) * blendmask[k]);
  }

  if(show_variance_mask)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < pixels*4; k +=4)
      out[k+3] = blendmask[k/4];

    error = FALSE;
    goto finalize;
  }

  gauss_idx = _cs_precalc_gauss_idx(self, roi, radius, d->cs_boost, d->cs_center);
  if(!gauss_idx) goto finalize;

  if(show_sigma_mask)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < pixels*4; k +=4)
      out[k+3] = (float)gauss_idx[k/4] / 255.0f;
    error = FALSE;
    goto finalize;
  }

  for(int iter = 0; iter < d->cs_iter && !dt_pipe_shutdown(pipe); iter++)
  {
    _blur_div(tmp1, tmp2, luminance, blendmask, gd->gauss_coeffs, gauss_idx, width, height);
    _blur_mul(tmp2, tmp1, blendmask, gd->gauss_coeffs, gauss_idx, width, height);
  }

  DT_OMP_FOR_SIMD()
  for(size_t k = 0; k < pixels; k++)
  {
    if(blendmask[k] > 0.0f)
    {
      const float luminance_new = interpolatef(CLIP(blendmask[k]), tmp1[k], luminance[k]);
      const float factor = luminance_new / MAX(luminance[k], CAPTURE_YMIN);
      for_each_channel(c) out[k*4 + c] *= factor;
    }
  }

  error = FALSE;

  finalize:
  if(error)
    dt_print_pipe(DT_DEBUG_ALWAYS, "capture sharpen failed", pipe, self, DT_DEVICE_CPU, NULL, NULL,
      "unable to allocate memory");

  dt_free_align(gauss_idx);
  dt_free_align(tmp2);
  dt_free_align(tmp1);
  dt_free_align(luminance);
  dt_free_align(blendmask);
}

// revert aggressive optimizing
#ifdef __GNUC__
  #pragma GCC pop_options
#endif

#if HAVE_OPENCL

int _capture_sharpen_cl(dt_iop_module_t *self,
                        dt_dev_pixelpipe_iop_t *const piece,
                        const cl_mem dev_in,
                        cl_mem dev_out,
                        const dt_iop_roi_t *const roi,
                        const gboolean showmask,
                        const gboolean show_sigmamask)
{
  dt_dev_pixelpipe_t *pipe = piece->pipe;

  const int width = roi->width;
  const int height = roi->height;
  const int pixels = width * height;
  const int bsize = sizeof(float) * pixels;
  const int devid = piece->pipe->devid;

  const dt_iop_demosaic_data_t *const d = piece->data;
  dt_iop_demosaic_global_data_t *const gd = self->global_data;
  dt_iop_demosaic_gui_data_t *g = self->gui_data;

  if(pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL)
  {
    const gboolean hqthumb = _get_thumb_quality(pipe->final_width, pipe->final_height);
    if(!hqthumb) return CL_SUCCESS;
  }

  if(d->cs_iter < 1 && !showmask) return CL_SUCCESS;

  const uint32_t filters = pipe->dsc.filters;
  const dt_iop_buffer_dsc_t *dsc = &pipe->dsc;
  const gboolean wbon = dsc->temperature.enabled;
  dt_aligned_pixel_t icoeffs = { wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[0] : CAPTURE_CFACLIP,
                                 wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[1] : CAPTURE_CFACLIP,
                                 wbon ? CAPTURE_CFACLIP * dsc->temperature.coeffs[2] : CAPTURE_CFACLIP,
                                 0.0f };

  const gboolean fullpipe = pipe->type & DT_DEV_PIXELPIPE_FULL;
  const gboolean autoradius = fullpipe && g && g->autoradius;
  const float old_radius = d->cs_radius;
  float radius = old_radius;
  if(autoradius || radius < 0.01f)
  {
    float *in = dt_alloc_align_float(pixels);
    if(in)
    {
      if(dt_opencl_copy_device_to_host(devid, in, dev_in, width, height, sizeof(float)) == CL_SUCCESS)
      {
        radius = filters != 9u
                ? _calcRadiusBayer(in, width, height, 0.01f, 1.0f, filters)
                : _calcRadiusXtrans(in, 0.01f, 1.0f, roi, (const uint8_t(*const)[6])pipe->dsc.xtrans);
        const gboolean valid = radius > 0.1f && radius < 1.0f;
        dt_print_pipe(DT_DEBUG_PIPE, filters != 9u ? "bayer autoradius" : "xtrans autoradius",
            pipe, self, devid, roi, NULL, "autoradius=%.2f", radius);

        if(!feqf(radius, old_radius, 0.005f) && valid)
        {
          if(fullpipe)
          {
            if(g)
            {
              dt_control_log(_("calculated radius"));
              g->autoradius = TRUE;
            }
            dt_iop_demosaic_params_t *p = self->params;
            p->cs_radius = radius;
          }
        }
        else if(g) g->autoradius = FALSE;
      }
      dt_free_align(in);
    }
  }

  cl_mem gcoeffs = NULL;
  cl_mem gauss_idx = NULL;

  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  cl_mem blendmask = dt_opencl_alloc_device_buffer(devid, bsize);
  cl_mem luminance = dt_opencl_alloc_device_buffer(devid, bsize);
  cl_mem tmp2 = dt_opencl_alloc_device_buffer(devid, bsize);
  cl_mem tmp1 = dt_opencl_alloc_device_buffer(devid, bsize);
  cl_mem xtrans = dt_opencl_copy_host_to_device_constant(devid, sizeof(pipe->dsc.xtrans), pipe->dsc.xtrans);
  cl_mem whites = dt_opencl_copy_host_to_device_constant(devid, 4 * sizeof(float), icoeffs);
  cl_mem dev_rgb = dt_opencl_duplicate_image(devid, dev_out);

  if(!blendmask || !luminance || !tmp2 || !tmp1 || !xtrans || !whites || !dev_rgb) goto finish;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->prefill_clip_mask, width, height,
          CLARG(tmp2), CLARG(width), CLARG(height));
  if(err != CL_SUCCESS) goto finish;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->prepare_blend, width, height,
          CLARG(dev_in), CLARG(dev_out), CLARG(filters), CLARG(xtrans), CLARG(tmp2), CLARG(tmp1),
          CLARG(whites), CLARG(width), CLARG(height));
  if(err != CL_SUCCESS) goto finish;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->modify_blend, width, height,
          CLARG(tmp2), CLARG(tmp1), CLARG(luminance), CLARG(d->cs_thrs), CLARG(width), CLARG(height));
  if(err != CL_SUCCESS) goto finish;

  err = dt_gaussian_fast_blur_cl_buffer(devid, tmp2, blendmask, width, height, 2.0f, 1, 0.0f, 1.0f);
  if(err != CL_SUCCESS) goto finish;

  err = dt_opencl_enqueue_kernel_1d_args(devid, gd->final_blend, pixels,
          CLARG(blendmask), CLARG(tmp2), CLARG(pixels));
  if(err != CL_SUCCESS) goto finish;

  if(showmask)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->show_blend_mask, width, height,
          CLARG(dev_rgb), CLARG(dev_out), CLARG(blendmask), CLARG(gauss_idx),
          CLARG(width), CLARG(height), CLARG(showmask));
    goto finish;
  }

  unsigned char *f_gauss_idx = _cs_precalc_gauss_idx(self, roi, radius, d->cs_boost, d->cs_center);
  if(f_gauss_idx)
  {
    gcoeffs = dt_opencl_copy_host_to_device_constant(devid, sizeof(float) * (UCHAR_MAX+1) * CAPTURE_KERNEL_ALIGN, gd->gauss_coeffs);
    gauss_idx = dt_opencl_copy_host_to_device_constant(devid, sizeof(unsigned char) * height * width, f_gauss_idx);
  }
  dt_free_align(f_gauss_idx);

  err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
  if(!gcoeffs || !gauss_idx) goto finish;

  if(show_sigmamask)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->show_blend_mask, width, height,
          CLARG(dev_rgb), CLARG(dev_out), CLARG(blendmask), CLARG(gauss_idx),
          CLARG(width), CLARG(height), CLARG(showmask));
    goto finish;
  }

  for(int iter = 0; iter < d->cs_iter && !dt_pipe_shutdown(pipe); iter++)
  {
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->gaussian_9x9_div, width, height,
      CLARG(tmp1), CLARG(tmp2), CLARG(luminance), CLARG(blendmask),
      CLARG(gcoeffs), CLARG(gauss_idx), CLARG(width), CLARG(height));
    if(err != CL_SUCCESS) goto finish;

    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->gaussian_9x9_mul, width, height,
      CLARG(tmp2), CLARG(tmp1), CLARG(blendmask),
      CLARG(gcoeffs), CLARG(gauss_idx), CLARG(width), CLARG(height));
    if(err != CL_SUCCESS) goto finish;
  }

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->capture_result, width, height,
    CLARG(dev_rgb), CLARG(dev_out), CLARG(blendmask), CLARG(luminance), CLARG(tmp1),
    CLARG(width), CLARG(height));

  finish:
  if(err != CL_SUCCESS)
    dt_print_pipe(DT_DEBUG_ALWAYS, "capture sharpen failed",
      pipe, self, devid, NULL, NULL,
      "Error: %s", cl_errstr(err));

  dt_opencl_release_mem_object(gauss_idx);
  dt_opencl_release_mem_object(gcoeffs);
  dt_opencl_release_mem_object(blendmask);
  dt_opencl_release_mem_object(dev_rgb);
  dt_opencl_release_mem_object(tmp2);
  dt_opencl_release_mem_object(tmp1);
  dt_opencl_release_mem_object(luminance);
  dt_opencl_release_mem_object(xtrans);
  dt_opencl_release_mem_object(whites);

  return err;
}
#endif // OpenCL
