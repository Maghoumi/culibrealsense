#pragma once
#ifndef LIBREALSENSE_CUDA_CONVERSION_H
#define LIBREALSENSE_CUDA_CONVERSION_H

#ifdef RS_USE_CUDA

// Types
#include <stdint.h>
#include "../include/librealsense/rs.h"
#include "assert.h"
#include "types.h"

// CUDA headers
#include <cuda_runtime.h>

#ifdef _MSC_VER 
// Add library dependencies if using VS
#pragma comment(lib, "cudart_static")
#endif

#define RS_CUDA_THREADS_PER_BLOCK 16

namespace rsimpl
{
	void unpack_yuy2_rgb8_cuda(const uint8_t* src, uint8_t* dst, int n);

	template<rs_format FORMAT> void unpack_yuy2_cuda(byte * const d[], const byte * s, int n)
	{
		const uint8_t *src = reinterpret_cast<const uint8_t *>(s);
		uint8_t *dst = reinterpret_cast<uint8_t *>(d[0]);

		switch (FORMAT)
		{
		case RS_FORMAT_Y8:
			break;
		case RS_FORMAT_Y16:
			break;
		case RS_FORMAT_RGB8:
			unpack_yuy2_rgb8_cuda(src, dst, n);
			break;
		case RS_FORMAT_BGR8:
			break;
		case RS_FORMAT_RGBA8:
			break;
		case RS_FORMAT_BGRA8:
			break;
		default:
			assert(false);
		}
	}
}

#endif // RS_USE_CUDA

#endif // LIBREALSENSE_CUDA_CONVERSION_H