#include "common/check.h"
#include "core/tensor.h"
#include "ops.h"
#include "registry.h"

namespace hpolib::ops {

Tensor abs(const Tensor& input) {
  HPOLIB_CHECK(input.numel() >= 0,
               "[Ops::abs]: the size should be greater or equal than 0.");
  Tensor output(input.shape(), input.options());
  output = ops::abs_dispatch(input.dataPtr());
  return output;
}

void abs(Tensor& output, const Tensor& input) {
  HPOLIB_CHECK(input.device() == output.device(), "input and output must be on the same device.");
  HPOLIB_CHECK(input.shape() == output.shape(), "output's shape must be euqal to input's.");
  ops::absOut_dispatch(output.dataPtr(), input.dataPtr());
}

void abs(Tensor& input) {
  HPOLIB_CHECK(input.numel() >= 0,
               "[Ops::abs]: the size should be greater or equal than 0.");
  ops::absInplace_dispatch(input.dataPtr());
}

}  // namespace hpolib::ops