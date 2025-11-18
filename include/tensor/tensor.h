#ifndef __COREFORGE_TENSOR_H__
#define __COREFORGE_TENSOR_H__

#include <memory>

#include "tensorImpl.h"
#include "scalar.h"

namespace coreforge { 
class Tensor {
  public:
  private:
    std::shared_ptr<TensorImpl> impl_;
};
} // namespace coreforge

#endif // __COREFORGE_TENSOR_H__