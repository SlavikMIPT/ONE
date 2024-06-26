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

#ifndef ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H

#include "PALUtils.h"
#include "core/OMKernelData.h"
#include "core/OMRuntimeShape.h"
#include "execute/OMUtils.h"
#include "OMStatus.h"
#include <cassert>
namespace onert_micro
{
namespace execute
{
namespace pal
{
namespace
{
struct GateParameters
{
  core::FullyConnectedParams input_fc_params;
  core::FullyConnectedParams recurrent_fc_params;
};

struct InterGateParameters
{
  core::ArithmeticParams forget_cell_mul_params;
  core::ArithmeticParams input_mul_params;
  core::ArithmeticParams output_mul_params;
};

struct CellStateInfo
{
  float cell_clip;
  // clipping range for cell state only 16 bits cell is supported (could be
  // generalized through templatation)
  int16_t quantized_cell_clip;
  // 2^-cell_state_scale_power = cell state scale, required by integer tanh
  // computation
  int32_t cell_state_scale_power;
};

struct LSTMParameters
{
  GateParameters forget_gate_parameters;
  GateParameters input_gate_parameters;
  GateParameters cell_gate_parameters;
  GateParameters output_gate_parameters;
  InterGateParameters inter_gate_parameters;
};

template <typename T> core::FullyConnectedParams createFCParameters()
{
  assert(false && "Not Implemented Yet");
}

template <> core::FullyConnectedParams createFCParameters<float>()
{
  core::FullyConnectedParams op_params{};

  execute::calculateActivationRange(circle::ActivationFunctionType::ActivationFunctionType_NONE,
                                    &op_params.float_activation_min,
                                    &op_params.float_activation_max);
  op_params.quantized_activation_max = static_cast<int32_t>(op_params.float_activation_max);
  op_params.quantized_activation_min = static_cast<int32_t>(op_params.float_activation_min);
  return op_params;
}

template <typename T> GateParameters createGateParameters()
{
  assert(false && "Not Implemented Yet");
}

template <> GateParameters createGateParameters<float>()
{
  GateParameters gate_params;
  gate_params.input_fc_params = createFCParameters<float>();
  gate_params.recurrent_fc_params = createFCParameters<float>();

  return gate_params;
}

template <typename T> void prepareGateParameters(LSTMParameters &lstm_params)
{
  assert(false && "Not Implemented Yet");
}

template <> void prepareGateParameters<float>(LSTMParameters &lstm_params)
{
  // Gate Parameters
  lstm_params.forget_gate_parameters = createGateParameters<float>();
  lstm_params.input_gate_parameters = createGateParameters<float>();
  lstm_params.cell_gate_parameters = createGateParameters<float>();
  lstm_params.output_gate_parameters = createGateParameters<float>();

  // Inter gate multiplication parameters
  core::ArithmeticParams op_params{};
  execute::calculateActivationRange(circle::ActivationFunctionType::ActivationFunctionType_NONE,
                                    &op_params.float_activation_min,
                                    &op_params.float_activation_max);
  op_params.quantized_activation_max = static_cast<int32_t>(op_params.float_activation_max);
  op_params.quantized_activation_min = static_cast<int32_t>(op_params.float_activation_min);
  lstm_params.inter_gate_parameters.forget_cell_mul_params = op_params;
  lstm_params.inter_gate_parameters.input_mul_params = op_params;
  lstm_params.inter_gate_parameters.output_mul_params = op_params;
}

} // namespace
template <typename T>
inline OMStatus UnidirectionalSequenceLSTM(const core::OMRuntimeShape &input_shape,
                                           const T *input_data,
                                           const circle::UnidirectionalSequenceLSTMOptions *options,
                                           const core::OMRuntimeShape &output_shape, T *output_data)
{
  assert(false && "Not Implemented Yet");
}

template <>
inline OMStatus
UnidirectionalSequenceLSTM<float>(const core::OMRuntimeShape &input_shape, const float *input_data,
                                  const circle::UnidirectionalSequenceLSTMOptions *options,
                                  const core::OMRuntimeShape &output_shape, float *output_data)
{
  CellStateInfo cell_state_info{options->cell_clip(), 0, 0}; // No quantization
  LSTMParameters lstm_params{};
  prepareGateParameters<float>(lstm_params);

  const bool time_major = options->time_major();
  const auto batch_size = time_major ? input_shape.dims(1) : input_shape.dims(0);
  const auto state_dimension = output_shape.dims(1);

  //  const auto cell_state_type_size =
  //  getDataTypeSize(Tensor::element_type(lstm_struct.cell_state()));
  return Ok;
}
} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H
