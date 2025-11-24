#ifndef __HPOLIB_REGISTRY_H__
#define __HPOLIB_REGISTRY_H__

#include "core/dispatch.h"
#include "core/tensor.h"

namespace hpolib {
namespace ops {

using UnaryOpFn = Tensor(const Tensor& input);
using UnaryOpOutFn = void(Tensor& output, const Tensor& input);
using UnaryOpInplaceFn = void(Tensor& input);

DEFINE_DISPATCH(abs, UnaryOpFn)
DEFINE_DISPATCH(absOut, UnaryOpOutFn)
DEFINE_DISPATCH(absInplace, UnaryOpInplaceFn)
}  // namespace ops
}  // namespace hpolib

#endif  // __HPOLIB_REGISTRY_H__