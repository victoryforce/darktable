/*
    This file is part of darktable,
    Copyright (C) 2012-2025 darktable developers.

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

#include "imageio_gm.h"
#include "common/colorspaces.h"
#include "common/darktable.h"
#include "common/exif.h"
#include "imageio_common.h"

#include <assert.h>
#include <inttypes.h>
#include <magick/api.h>
#include <memory.h>
#include <stdio.h>
#include <strings.h>


// We only support images with certain filename extensions via GraphicsMagick.
// RAWs are excluded as GraphicsMagick would render them with third party
// libraries in reduced quality - slow and only 8-bit.
static gboolean _supported_image(const gchar *filename)
{
  const char *extensions_whitelist[] =
    {
    "tiff", "tif", // may support more format features than the native loader
    "pam", "pbm", "pgm", "ppm", "pnm", // PNM-style formats
    "jpc", "jp2", "jpf", "jpx", // JPEG 2000 extensions
    "bmp", "miff", "dcm", "jng", "mng", "gif",  // misc exotic formats
    "fits", "fit", "fts",  // FITS format (common in astro imagery)
    "cin", "dpx", // Kodak made formats used in film industry for still frames
    "jxl",  // probably of no practical use
    "webp", // probably of no practical use
    NULL
    };

  gboolean supported = FALSE;
  char *ext = g_strrstr(filename, ".");
  if(!ext) return FALSE;
  ext++;
  for(const char **i = extensions_whitelist; *i != NULL; i++)
    if(!g_ascii_strncasecmp(ext, *i, strlen(*i)))
    {
      supported = TRUE;
      break;
    }
  return supported;
}


dt_imageio_retval_t dt_imageio_open_gm(dt_image_t *img,
                                       const char *filename,
                                       dt_mipmap_buffer_t *mbuf)
{
  int err = DT_IMAGEIO_LOAD_FAILED;

  if(!_supported_image(filename))
    return DT_IMAGEIO_LOAD_FAILED;

  if(!img->exif_inited)
    (void)dt_exif_read(img, filename);

  ExceptionInfo exception;
  GetExceptionInfo(&exception);
  ImageInfo *image_info = CloneImageInfo((ImageInfo *)NULL);

  g_strlcpy(image_info->filename, filename, sizeof(image_info->filename));

  Image *image = ReadImage(image_info, &exception);

  if(exception.severity != UndefinedException)
    CatchException(&exception);

  if(!image)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[GraphicsMagick_open] ReadImage failed for '%s'",
             img->filename);
    err = DT_IMAGEIO_FILE_NOT_FOUND;
    goto error;
  }

  dt_print(DT_DEBUG_IMAGEIO,
           "[GraphicsMagick_open] loading image '%s'",
           img->filename);

  img->width = image->columns;
  img->height = image->rows;

  img->buf_dsc.channels = 4;
  img->buf_dsc.datatype = TYPE_FLOAT;

  float *mipbuf = (float *)dt_mipmap_cache_alloc(mbuf, img);
  if(!mipbuf)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[GraphicsMagick_open] could not alloc full buffer for image '%s'",
             img->filename);
    err = DT_IMAGEIO_CACHE_FULL;
    goto error;
  }

  char *colormap;
  if(IsCMYKColorspace(image->colorspace))
    colormap = "CMYK";
  else
    colormap = "RGBP";

  int ret = DispatchImage(image,
                          0,
                          0,
                          img->width,
                          img->height,
                          colormap,
                          FloatPixel,
                          mipbuf,
                          &exception);

  if(exception.severity != UndefinedException)
    CatchException(&exception);

  if(ret != MagickPass)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[GraphicsMagick_open] error reading image pixels for '%s'",
             img->filename);
    err = DT_IMAGEIO_LOAD_FAILED;
    goto error;
  }

  // If the image in CMYK color space convert it to linear RGB
  if(IsCMYKColorspace(image->colorspace))
  {
    for(size_t index = 0; index < img->width * img->height * 4; index += 4)
    {
      float black = mipbuf[index + 3];
      mipbuf[index]     = (1.f - black) * (1.f - mipbuf[index]);
      mipbuf[index + 1] = (1.f - black) * (1.f - mipbuf[index + 1]);
      mipbuf[index + 2] = (1.f - black) * (1.f - mipbuf[index + 2]);
    }
  }

  size_t profile_length;
  const uint8_t *profile_data;
  profile_data = (const uint8_t *)GetImageProfile(image,
                                                  "ICM",
                                                  &profile_length);

  if(profile_data)
  {
    img->profile = g_try_malloc0(profile_length);
    if(img->profile)
    {
      memcpy(img->profile, profile_data, profile_length);
      img->profile_size = profile_length;
    }
  }

  if(image)
    DestroyImage(image);

  if(image_info)
    DestroyImageInfo(image_info);

  DestroyExceptionInfo(&exception);

  img->buf_dsc.cst = IOP_CS_RGB;
  img->buf_dsc.filters = 0u;
  img->flags &= ~DT_IMAGE_RAW;
  img->flags &= ~DT_IMAGE_HDR;
  img->flags &= ~DT_IMAGE_S_RAW;
  img->flags |= DT_IMAGE_LDR;

  img->loader = LOADER_GM;
  return DT_IMAGEIO_OK;

error:
  if(image)
    DestroyImage(image);

  if(image_info)
    DestroyImageInfo(image_info);

  DestroyExceptionInfo(&exception);

  return err;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
