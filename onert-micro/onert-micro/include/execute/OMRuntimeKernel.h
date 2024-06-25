/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_EXECUTE_RUNTIME_KERNEL_H
#define ONERT_MICRO_EXECUTE_RUNTIME_KERNEL_H

#include "core/reader/OMCircleReader.h"
#include "core/OMRuntimeContext.h"
#include "core/OMRuntimeStorage.h"

#include <cstdint>

namespace onert_micro
{
namespace execute
{
template <uint32_t maxInputSize, uint32_t maxOutputSize> class OMBaseRuntimeKernel
{
public:
  OMBaseRuntimeKernel() = default;
  OMBaseRuntimeKernel(const OMBaseRuntimeKernel &) = delete;
  OMBaseRuntimeKernel(OMBaseRuntimeKernel &&) = delete;
  ~OMBaseRuntimeKernel() = default;
  OMBaseRuntimeKernel &operator=(const OMBaseRuntimeKernel &) = delete;
  OMBaseRuntimeKernel &&operator=(const OMBaseRuntimeKernel &&) = delete;

public:
  OMStatus readKernel(uint16_t op_index, core::OMRuntimeContext &runtime_context)
  {
    {
      first_operator = runtime_context.getCircleOperatorAt(op_index);
      const circle::Operator *last_operator = runtime_context.getCircleOperatorAt(op_index);

      inputs_num = first_operator->inputs()->size();
      assert(inputs_num < maxInputSize);

      if (inputs_num >= maxInputSize)
        return UnknownError;

      outputs_num = last_operator->outputs()->size();
      assert(outputs_num < maxOutputSize);

      if (outputs_num >= maxOutputSize)
        return UnknownError;

      assert(inputs_num > 0 and outputs_num > 0);

      // Read inputs
      {
        const auto *inputs_op = first_operator->inputs();
        for (uint32_t i = 0; i < inputs_num; ++i)
        {
          inputs_index[i] = inputs_op->operator[](i);
          if (inputs_index[i] != -1)
            inputs[i] = runtime_context.getTensorByIndex(inputs_index[i]);
        }
      }
      // Read outputs
      {
        const auto *outputs_op = last_operator->outputs();
        for (uint32_t i = 0; i < outputs_num; ++i)
        {
          outputs_index[i] = outputs_op->operator[](i);
          if (outputs_index[i] != -1)
            outputs[i] = runtime_context.getTensorByIndex(outputs_index[i]);
        }
      }

      return Ok;
    }
  }

  OMStatus getDataFromStorage(uint16_t op_index, core::OMRuntimeStorage &storage,
                              core::OMRuntimeContext &context)
  {
    {
      OMStatus status = Ok;

      for (uint32_t i = 0; i < inputs_num; ++i)
      {
        if (inputs_index[i] == -1)
          continue;
        status = storage.getDataByTensorIndex(&inputs_data[i], inputs_index[i]);
        if (inputs_data[i] == nullptr)
          status = context.getConstDataByTensorIndex(&inputs_data[i], inputs_index[i]);
        if (status != Ok)
          return status;
      }

      for (uint32_t i = 0; i < outputs_num; ++i)
      {
        if (outputs_index[i] == -1)
          continue;
        status = storage.getDataByTensorIndex(&outputs_data[i], outputs_index[i]);

        if (status != Ok)
          return status;

        if (storage.getKernelType(op_index) == core::Inplace)
        {
          outputs_data[i] = inputs_data[i];
          status = storage.removeTensorFromTensorIndexToData(inputs_index[i]);

          if (status != Ok)
            return status;

          status = storage.saveDataToTensorIndex(outputs_data[i], outputs_index[i]);
        }
      }

      return status;
    }
  }

public:
  const circle::Tensor *inputs[maxInputSize] = {nullptr};
  const circle::Tensor *outputs[maxOutputSize] = {nullptr};

  uint8_t *inputs_data[maxInputSize] = {nullptr};
  uint8_t *outputs_data[maxOutputSize] = {nullptr};

  int32_t inputs_index[maxInputSize] = {-1};
  int32_t outputs_index[maxOutputSize] = {-1};

  uint32_t outputs_num = -1;
  uint32_t inputs_num = -1;

  const circle::Operator *first_operator = nullptr;
};
class OMRuntimeKernel : public OMBaseRuntimeKernel<5, 5>
{
};
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_RUNTIME_KERNEL_H
