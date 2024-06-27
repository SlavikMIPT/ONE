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
  const circle::Tensor *cell_state = nullptr;
  const circle::Tensor *activation_state = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data = nullptr;
  uint8_t *cell_state_data = nullptr;
  uint8_t *output_data = nullptr;

  OMStatus status = Ok;
  const circle::UnidirectionalSequenceLSTMOptions *options = nullptr;
  execute::OMLSTMRuntimeKernel runtime_kernel;
  {

    runtime_kernel.readKernel(op_index, runtime_context);

    input = runtime_kernel.inputs[kInputTensorIdx];
    cell_state = runtime_kernel.inputs[kCellStateTensorIdx];
    activation_state = runtime_kernel.inputs[kActivationStateTensorIdx];
    output = runtime_kernel.outputs[kOutputTensorIdx];

    assert(input != nullptr);
    assert(cell_state != nullptr);
    assert(activation_state != nullptr);
    assert(output != nullptr);

    status = runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);
    if (status != Ok)
      return status;

    input_data = runtime_kernel.inputs_data[kInputTensorIdx];
    cell_state_data = runtime_kernel.inputs_data[kCellStateTensorIdx];
    output_data = runtime_kernel.outputs_data[kOutputTensorIdx];
  }
  execute::pal::LSTMStruct lstm_struct(runtime_kernel);

  assert(input_data != nullptr);
  assert(cell_state_data != nullptr);
  assert(output_data != nullptr);

  switch (input->type())
  {
#ifndef DIS_FLOAT

    case circle::TensorType_FLOAT32:
      status = pal::UnidirectionalSequenceLSTM<float>(
        core::OMRuntimeShape(input), core::utils::castInputData<float>(input_data),
        core::OMRuntimeShape(cell_state), cell_state_data,
        core::onertMicroDatatype(cell_state->type()), core::OMRuntimeShape(activation_state),
        lstm_struct, core::OMRuntimeShape(output),
        core::utils::castOutputData<float>(output_data));
      break;
#endif // DIS_FLOAT
    default:
    {
      status = UnsupportedType;
      assert(false && "Unsupported type.");
    }
  }

  return status;
}
