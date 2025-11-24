#ifndef __HPOLIB_CONFIG_H__
#define __HPOLIB_CONFIG_H__

#if defined(__CUDACC__)
#define CF_HOST_DEVICE __host__ __device__
#else
#define CF_HOST_DEVICE
#endif

#endif  // __HPOLIB_CONFIG_H__
