#ifndef __HPOLIB_OPS_H__
#define __HPOLIB_OPS_H__

#include "core/tensor.h"

namespace hpolib {
namespace ops {
// abs
Tensor abs(const Tensor& input);
void abs(Tensor& output, const Tensor& input);
void abs(Tensor& input);

}  // namespace ops
}  // namespace hpolib

#endif  //__HPOLIB_OPS_H__