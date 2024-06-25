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

#include "import/OMKernelConfigureBuilder.h"
#include "core/OMUtils.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"

#include <algorithm>
#include <array>
using namespace onert_micro;
using namespace onert_micro::core;
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

class OMLSTMRuntimeKernel : public execute::OMBaseRuntimeKernel<24, 5>
{
};
} // namespace

OMStatus onert_micro::import::configure_kernel_CircleUnidirectionalSequenceLSTM(
  const OMConfigureArgs &config_args)
{
  OMRuntimeContext &runtime_context = config_args.runtime_context;
  uint16_t op_index = config_args.kernel_index;

  OMLSTMRuntimeKernel runtime_kernel;

  OMStatus status = runtime_kernel.readKernel(op_index, runtime_context);
  if (status != Ok)
    return status;

  const auto input = runtime_kernel.inputs[kInputTensorIdx];
  const auto input_to_input_weights = runtime_kernel.inputs[kInputToInputWeightsTensorIdx];
  const auto input_to_forget_weights = runtime_kernel.inputs[kInputToForgetWeightsTensorIdx];
  const auto input_to_cell_weights = runtime_kernel.inputs[kInputToCellWeightsTensorIdx];
  const auto input_to_output_weights = runtime_kernel.inputs[kInputToOutputWeightsTensorIdx];
  assert(input != nullptr);
  // input_to_input_weights - optional
  assert(input_to_forget_weights != nullptr);
  assert(input_to_cell_weights != nullptr);
  assert(input_to_output_weights != nullptr);

  const auto recurrent_to_input_weights = runtime_kernel.inputs[kRecurrentToInputWeightsTensorIdx];
  const auto recurrent_to_forget_weights =
    runtime_kernel.inputs[kRecurrentToForgetWeightsTensorIdx];
  const auto recurrent_to_cell_weights = runtime_kernel.inputs[kRecurrentToCellWeightsTensorIdx];
  const auto recurrent_to_output_weights =
    runtime_kernel.inputs[kRecurrentToOutputWeightsTensorIdx];
  assert(recurrent_to_input_weights != nullptr);
  // recurrent_to_input_weights - optional
  assert(recurrent_to_forget_weights != nullptr);
  assert(recurrent_to_cell_weights != nullptr);
  assert(recurrent_to_output_weights != nullptr);

  const auto cell_to_input_weights = runtime_kernel.inputs[kCellToInputWeightsTensorIdx];
  const auto cell_to_forget_weights = runtime_kernel.inputs[kCellToForgetWeightsTensorIdx];
  const auto cell_to_output_weights = runtime_kernel.inputs[kCellToOutputWeightsTensorIdx];
  // optional cell_to_input_weights
  // optional cell_to_forget_weights
  // optional cell_to_output_weights

  const auto input_gate_bias = runtime_kernel.inputs[kInputGateBiasTensorIdx];
  const auto forget_gate_bias = runtime_kernel.inputs[kForgetGateBiasTensorIdx];
  const auto cell_gate_bias = runtime_kernel.inputs[kCellGateBiasTensorIdx];
  const auto output_gate_bias = runtime_kernel.inputs[kOutputGateBiasTensorIdx];
  // optional input_gate_bias
  assert(forget_gate_bias != nullptr);
  assert(cell_gate_bias != nullptr);
  assert(output_gate_bias != nullptr);

  const auto projection_weights = runtime_kernel.inputs[kProjectionWeightsTensorIdx];
  const auto projection_bias = runtime_kernel.inputs[kProjectionBiasTensorIdx];
  // optional projection_weights
  // optional projection_bias

  const auto activation_state = runtime_kernel.inputs[kActivationStateTensorIdx];
  const auto cell_state = runtime_kernel.inputs[kCellStateTensorIdx];
  assert(activation_state != nullptr);
  assert(cell_state != nullptr);

  const auto input_layer_norm_coefficients =
    runtime_kernel.inputs[kInputLayerNormCoefficientsTensorIdx];
  const auto forget_layer_norm_coefficients =
    runtime_kernel.inputs[kForgetLayerNormCoefficientsTensorIdx];
  const auto cell_layer_norm_coefficients =
    runtime_kernel.inputs[kCellLayerNormCoefficientsTensorIdx];
  const auto output_layer_norm_coefficients =
    runtime_kernel.inputs[kOutputLayerNormCoefficientsTensorIdx];
  // optional input_layer_norm_coefficients
  // optional forget_layer_norm_coefficients
  // optional cell_layer_norm_coefficients
  // optional output_layer_norm_coefficients

  const auto output = runtime_kernel.outputs[kOutputTensorIdx];
  assert(output != nullptr);

  // Validate tensor types
  {
    status = utils::checkCondition(input->type() == activation_state->type());
    if (status != Ok)
      return status;

    status = utils::checkCondition(input->type() == output->type());
    if (status != Ok)
      return status;
  }

  {
    const std::array<const circle::Tensor *, 8> tmp_array{
      input_to_input_weights,    input_to_forget_weights,    input_to_cell_weights,
      input_to_output_weights,   recurrent_to_input_weights, recurrent_to_forget_weights,
      recurrent_to_cell_weights, recurrent_to_output_weights};
    for (auto &item : tmp_array)
    {
      status =
        utils::checkCondition(item == nullptr or item->type() == input_to_forget_weights->type());
      if (status != Ok)
        return status;
    }
  }

  {
    const std::array<const circle::Tensor *, 4> tmp_array{input_gate_bias, forget_gate_bias,
                                                          cell_gate_bias, output_gate_bias};
    for (const auto &item : tmp_array)
    {
      status = utils::checkCondition(item == nullptr or item->type() == forget_gate_bias->type());
      if (status != Ok)
        return status;
    }
  }
  return status;
}
