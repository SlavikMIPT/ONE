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

#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMUtils.h"
#include "PALUnidirectionalSequenceLSTM.h"

using namespace onert_micro;
using namespace onert_micro::execute;

namespace
{
constexpr int kInputTensorIdx = 0;
constexpr int kInputToInputWeightsTensorIdx = 1;
constexpr int kInputToForgetWeightsTensorIdx = 2;
constexpr int kInputToCellWeightsTensorIdx = 3;
constexpr int kInputToOutputWeightsTensorIdx = 4;
constexpr int kRecurrentToInputWeightsTensorIdx = 5;
constexpr int kRecurrentToForgetWeightsTensorIdx = 6;
constexpr int kRecurrentToCellWeightsTensorIdx = 7;
constexpr int kRecurrentToOutputWeightsTensorIdx = 8;
constexpr int kCellToInputWeightsTensorIdx = 9;
constexpr int kCellToForgetWeightsTensorIdx = 10;
constexpr int kCellToOutputWeightsTensorIdx = 11;
constexpr int kInputGateBiasTensorIdx = 12;
constexpr int kForgetGateBiasTensorIdx = 13;
constexpr int kCellGateBiasTensorIdx = 14;
constexpr int kOutputGateBiasTensorIdx = 15;
constexpr int kProjectionWeightsTensorIdx = 16;
constexpr int kProjectionBiasTensorIdx = 17;
constexpr int kActivationStateTensorIdx = 18;
constexpr int kCellStateTensorIdx = 19;
constexpr int kInputLayerNormCoefficientsTensorIdx = 20;
constexpr int kForgetLayerNormCoefficientsTensorIdx = 21;
constexpr int kCellLayerNormCoefficientsTensorIdx = 22;
constexpr int kOutputLayerNormCoefficientsTensorIdx = 23;

constexpr int kOutputTensorIdx = 0;

constexpr uint32_t kMaxInputSize = 24;
constexpr uint32_t kMaxOutputSize = 5;

} // namespace
execute::pal::CellStateInfo buildLstmCellStateInfoFloat(const float cell_clip)
{
  execute::pal::CellStateInfo cell_state_info;
  cell_state_info.cell_clip = cell_clip;
  cell_state_info.cell_state_scale_power = 0; // no quantization
  cell_state_info.quantized_cell_clip = 0;    // no quantization
  return cell_state_info;
}
OMStatus onert_micro::execute::execute_kernel_CircleUnidirectionalSequenceLSTM(
  const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  const circle::Tensor *input = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data = nullptr;
  uint8_t *output_data = nullptr;

  OMStatus status = Ok;
  const circle::UnidirectionalSequenceLSTMOptions *options = nullptr;

  {
    execute::OMBaseRuntimeKernel<kMaxInputSize, kMaxOutputSize> runtime_kernel;
    runtime_kernel.readKernel(op_index, runtime_context);

    input = runtime_kernel.inputs[kInputTensorIdx];
    output = runtime_kernel.outputs[kOutputTensorIdx];

    assert(input != nullptr);
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data = runtime_kernel.inputs_data[kInputTensorIdx];
    output_data = runtime_kernel.outputs_data[kOutputTensorIdx];
    options = runtime_kernel.first_operator->builtin_options_as_UnidirectionalSequenceLSTMOptions();
  }

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  switch (input->type())
  {
#ifndef DIS_FLOAT

    case circle::TensorType_FLOAT32:
      status = pal::UnidirectionalSequenceLSTM<float>(
        core::OMRuntimeShape(input), core::utils::castInputData<float>(input_data), options,
        core::OMRuntimeShape(output), core::utils::castOutputData<float>(output_data));
      break;
#endif // DIS_FLOAT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return Ok;
}
