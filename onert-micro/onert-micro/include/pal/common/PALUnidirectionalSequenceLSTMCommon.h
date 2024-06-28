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
#include "core/OMDataType.h"
#include "core/OMKernelData.h"
#include "core/OMRuntimeGraph.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMRuntimeShape.h"
#include "execute/OMUtils.h"
#include "OMStatus.h"

#include "PALTanh.h"
#include "PALMul.h"
#include "PALFullyConnected.h"

#include <cassert>

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
} // namespace

namespace onert_micro
{
namespace execute
{
namespace pal
{
struct LSTMStruct
{
  LSTMStruct() = delete;
  LSTMStruct(const LSTMStruct &) = delete;

  explicit LSTMStruct(const OMLSTMRuntimeKernel &runtime_kernel)
  {
    OMStatus status = Ok;
    _input = runtime_kernel.inputs[kInputTensorIdx];
    _input_to_input_weights = runtime_kernel.inputs[kInputToInputWeightsTensorIdx];
    _input_to_forget_weights = runtime_kernel.inputs[kInputToForgetWeightsTensorIdx];
    _input_to_cell_weights = runtime_kernel.inputs[kInputToCellWeightsTensorIdx];
    _input_to_output_weights = runtime_kernel.inputs[kInputToOutputWeightsTensorIdx];
    assert(_input != nullptr);
    (void)_input_to_input_weights; // optional
    assert(_input_to_forget_weights != nullptr);
    assert(_input_to_cell_weights != nullptr);
    assert(_input_to_output_weights != nullptr);

    _recurrent_to_input_weights = runtime_kernel.inputs[kRecurrentToInputWeightsTensorIdx];
    _recurrent_to_forget_weights = runtime_kernel.inputs[kRecurrentToForgetWeightsTensorIdx];
    _recurrent_to_cell_weights = runtime_kernel.inputs[kRecurrentToCellWeightsTensorIdx];
    _recurrent_to_output_weights = runtime_kernel.inputs[kRecurrentToOutputWeightsTensorIdx];
    assert(_recurrent_to_input_weights != nullptr);
    (void)_recurrent_to_input_weights; // optional
    assert(_recurrent_to_forget_weights != nullptr);
    assert(_recurrent_to_cell_weights != nullptr);
    assert(_recurrent_to_output_weights != nullptr);

    _cell_to_input_weights = runtime_kernel.inputs[kCellToInputWeightsTensorIdx];
    _cell_to_forget_weights = runtime_kernel.inputs[kCellToForgetWeightsTensorIdx];
    _cell_to_output_weights = runtime_kernel.inputs[kCellToOutputWeightsTensorIdx];
    (void)_cell_to_input_weights;  // optional
    (void)_cell_to_forget_weights; // optional
    (void)_cell_to_output_weights; // optional

    _input_gate_bias = runtime_kernel.inputs[kInputGateBiasTensorIdx];
    _forget_gate_bias = runtime_kernel.inputs[kForgetGateBiasTensorIdx];
    _cell_gate_bias = runtime_kernel.inputs[kCellGateBiasTensorIdx];
    _output_gate_bias = runtime_kernel.inputs[kOutputGateBiasTensorIdx];
    (void)_input_gate_bias; // optional
    assert(_forget_gate_bias != nullptr);
    assert(_cell_gate_bias != nullptr);
    assert(_output_gate_bias != nullptr);

    _projection_weights = runtime_kernel.inputs[kProjectionWeightsTensorIdx];
    _projection_bias = runtime_kernel.inputs[kProjectionBiasTensorIdx];
    (void)_projection_weights; // optional
    (void)_projection_bias;    // optional

    _activation_state = runtime_kernel.inputs[kActivationStateTensorIdx];
    _cell_state = runtime_kernel.inputs[kCellStateTensorIdx];
    assert(_activation_state != nullptr);
    assert(_cell_state != nullptr);

    _input_layer_norm_coefficients = runtime_kernel.inputs[kInputLayerNormCoefficientsTensorIdx];
    _forget_layer_norm_coefficients = runtime_kernel.inputs[kForgetLayerNormCoefficientsTensorIdx];
    _cell_layer_norm_coefficients = runtime_kernel.inputs[kCellLayerNormCoefficientsTensorIdx];
    _output_layer_norm_coefficients = runtime_kernel.inputs[kOutputLayerNormCoefficientsTensorIdx];
    (void)_input_layer_norm_coefficients;  // optional
    (void)_forget_layer_norm_coefficients; // optional
    (void)_cell_layer_norm_coefficients;   // optional
    (void)_output_layer_norm_coefficients; // optional

    _output = runtime_kernel.outputs[kOutputTensorIdx];
    assert(_output != nullptr);

    // Validate tensor types
    {
      status = core::utils::checkCondition(_input->type() == _activation_state->type());
      if (status != Ok)
      {
        _is_valid = false;
        return;
      }

      status = core::utils::checkCondition(_input->type() == _output->type());
      if (status != Ok)
      {
        _is_valid = false;
        return;
      }
    }

    {
      const std::array<const circle::Tensor *, 8> tmp_array{
        _input_to_input_weights,    _input_to_forget_weights,    _input_to_cell_weights,
        _input_to_output_weights,   _recurrent_to_input_weights, _recurrent_to_forget_weights,
        _recurrent_to_cell_weights, _recurrent_to_output_weights};
      for (auto &item : tmp_array)
      {
        status = core::utils::checkCondition(item == nullptr or
                                             item->type() == _input_to_forget_weights->type());
        if (status != Ok)
        {
          _is_valid = false;
          return;
        }
      }
    }

    {
      const std::array<const circle::Tensor *, 4> tmp_array{_input_gate_bias, _forget_gate_bias,
                                                            _cell_gate_bias, _output_gate_bias};
      for (const auto &item : tmp_array)
      {
        status =
          core::utils::checkCondition(item == nullptr or item->type() == _forget_gate_bias->type());
        if (status != Ok)
        {
          _is_valid = false;
          return;
        }
      }
    }

    _options =
      runtime_kernel.first_operator->builtin_options_as_UnidirectionalSequenceLSTMOptions();
    assert(_options != nullptr);

    _is_valid = true;
  }
  bool is_valid(void) { return _is_valid; }

  const circle::Tensor *input() const { return _input; }
  const circle::Tensor *input_to_input_weights() const { return _input_to_input_weights; }
  const circle::Tensor *input_to_forget_weights() const { return _input_to_forget_weights; }
  const circle::Tensor *input_to_cell_weights() const { return _input_to_cell_weights; }
  const circle::Tensor *input_to_output_weights() const { return _input_to_output_weights; }
  const circle::Tensor *recurrent_to_input_weights() const { return _recurrent_to_input_weights; }
  const circle::Tensor *recurrent_to_forget_weights() const { return _recurrent_to_forget_weights; }
  const circle::Tensor *recurrent_to_cell_weights() const { return _recurrent_to_cell_weights; }
  const circle::Tensor *recurrent_to_output_weights() const { return _recurrent_to_output_weights; }
  const circle::Tensor *cell_to_input_weights() const { return _cell_to_input_weights; }
  const circle::Tensor *cell_to_forget_weights() const { return _cell_to_forget_weights; }
  const circle::Tensor *cell_to_output_weights() const { return _cell_to_output_weights; }
  const circle::Tensor *input_gate_bias() const { return _input_gate_bias; }
  const circle::Tensor *forget_gate_bias() const { return _forget_gate_bias; }
  const circle::Tensor *cell_gate_bias() const { return _cell_gate_bias; }
  const circle::Tensor *output_gate_bias() const { return _output_gate_bias; }
  const circle::Tensor *projection_weights() const { return _projection_weights; }
  const circle::Tensor *projection_bias() const { return _projection_bias; }
  const circle::Tensor *activation_state() const { return _activation_state; }
  const circle::Tensor *cell_state() const { return _cell_state; }
  const circle::Tensor *input_layer_norm_coefficients() const
  {
    return _input_layer_norm_coefficients;
  }
  const circle::Tensor *forget_layer_norm_coefficients() const
  {
    return _forget_layer_norm_coefficients;
  }
  const circle::Tensor *cell_layer_norm_coefficients() const
  {
    return _cell_layer_norm_coefficients;
  }
  const circle::Tensor *output_layer_norm_coefficients() const
  {
    return _output_layer_norm_coefficients;
  }
  const circle::Tensor *output() const { return _output; }

  const circle::UnidirectionalSequenceLSTMOptions *options() const { return _options; };

  const uint8_t *input_data = nullptr;
  const uint8_t *input_to_forget_weights_data = nullptr;
  const uint8_t *input_to_input_weights_data = nullptr;
  const uint8_t *input_to_cell_weights_data = nullptr;

  const uint8_t *forget_gate_bias_data = nullptr;
  const uint8_t *input_gate_bias_data = nullptr;
  const uint8_t *cell_gate_bias_data = nullptr;

  const uint8_t *recurrent_to_forget_weights_data = nullptr;
  const uint8_t *recurrent_to_input_weights_data = nullptr;
  const uint8_t *recurrent_to_cell_weights_data = nullptr;

  const uint8_t *activation_state_data = nullptr;

private:
  const circle::Tensor *_input = nullptr;
  const circle::Tensor *_input_to_input_weights = nullptr;
  const circle::Tensor *_input_to_forget_weights = nullptr;
  const circle::Tensor *_input_to_cell_weights = nullptr;
  const circle::Tensor *_input_to_output_weights = nullptr;
  const circle::Tensor *_recurrent_to_input_weights = nullptr;
  const circle::Tensor *_recurrent_to_forget_weights = nullptr;
  const circle::Tensor *_recurrent_to_cell_weights = nullptr;
  const circle::Tensor *_recurrent_to_output_weights = nullptr;
  const circle::Tensor *_cell_to_input_weights = nullptr;
  const circle::Tensor *_cell_to_forget_weights = nullptr;
  const circle::Tensor *_cell_to_output_weights = nullptr;
  const circle::Tensor *_input_gate_bias = nullptr;
  const circle::Tensor *_forget_gate_bias = nullptr;
  const circle::Tensor *_cell_gate_bias = nullptr;
  const circle::Tensor *_output_gate_bias = nullptr;
  const circle::Tensor *_projection_weights = nullptr;
  const circle::Tensor *_projection_bias = nullptr;
  const circle::Tensor *_activation_state = nullptr;
  const circle::Tensor *_cell_state = nullptr;
  const circle::Tensor *_input_layer_norm_coefficients = nullptr;
  const circle::Tensor *_forget_layer_norm_coefficients = nullptr;
  const circle::Tensor *_cell_layer_norm_coefficients = nullptr;
  const circle::Tensor *_output_layer_norm_coefficients = nullptr;
  const circle::Tensor *_output = nullptr;
  const circle::UnidirectionalSequenceLSTMOptions *_options = nullptr;

  bool _is_valid = false;
};

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

template <typename T>
inline void addElementWise(const T *input_1, const T *input_2, int n_batch, int n_input, T *output)
{
  assert(false && "Not Implemented Yet");
}

template <>
inline void addElementWise<float>(const float *input_1, const float *input_2, int n_batch,
                                  int n_input, float *output)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    for (int i = 0; i < n_input; ++i)
    {
      const int index = batch * n_input + i;
      output[index] = input_1[index] + input_2[index];
    }
  }
}

template <typename T> inline core::FullyConnectedParams createFCParameters()
{
  assert(false && "Not Implemented Yet");
}

template <> inline core::FullyConnectedParams createFCParameters<float>()
{
  core::FullyConnectedParams op_params{};

  execute::calculateActivationRange(circle::ActivationFunctionType::ActivationFunctionType_NONE,
                                    &op_params.float_activation_min,
                                    &op_params.float_activation_max);
  op_params.quantized_activation_max = static_cast<int32_t>(op_params.float_activation_max);
  op_params.quantized_activation_min = static_cast<int32_t>(op_params.float_activation_min);
  return op_params;
}

template <typename T> inline GateParameters createGateParameters()
{
  assert(false && "Not Implemented Yet");
}

template <> inline GateParameters createGateParameters<float>()
{
  GateParameters gate_params;
  gate_params.input_fc_params = createFCParameters<float>();
  gate_params.recurrent_fc_params = createFCParameters<float>();

  return gate_params;
}

template <typename T> inline void prepareGateParameters(LSTMParameters &lstm_params)
{
  assert(false && "Not Implemented Yet");
}

template <> inline void prepareGateParameters<float>(LSTMParameters &lstm_params)
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

// Size information about the LSTM kernel, which is deduced from tensors stored
// in the flat buffer file.
struct LstmSizeInfo
{
  bool time_major;
  int32_t batch_size;
  int32_t time_steps;
  int32_t input_dimension;
  int32_t state_dimension;
};

class LstmStepManager
{
public:
  LstmStepManager() = delete;
  // Does not take any ownership, and all pointers must refer to valid objects
  // that outlive the one constructed.
  explicit LstmStepManager(const LstmSizeInfo &size_info) : size_info_(size_info) {}

  void updateTime()
  {
    current_time_ += 1;
    // default as one batch per inference
    int input_step = size_info_.input_dimension;
    int output_step = size_info_.state_dimension;
    // time major: batch inference
    if (size_info_.time_major)
    {
      input_step = input_step * size_info_.batch_size;
      output_step = output_step * size_info_.batch_size;
    }

    input_offset_ += input_step;
    output_offset_ += output_step;
  }

  void updateBatch()
  {
    current_batch_ += 1;
    // batch inference for time major: no action needed
    if (size_info_.time_major)
    {
      return;
    }
    // otherwise: singe batch inference, go to the next batch
    hidden_state_offset_ += size_info_.state_dimension;
    cell_state_offset_ += size_info_.state_dimension;
  }

  void resetTime() { current_time_ = 0; }

  core::OMRuntimeShape inputShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.input_dimension};
    const auto *dims_data = reinterpret_cast<const int32_t *>(dims);
    return {2, dims_data};
  }

  core::OMRuntimeShape stateShape() const
  {
    int batch_size = 1;
    if (size_info_.time_major)
    {
      batch_size = size_info_.batch_size;
    }
    const int dims[2] = {batch_size, size_info_.state_dimension};
    const auto *dims_data = reinterpret_cast<const int32_t *>(dims);
    return {2, dims_data};
  }

  int inputOffset() const { return input_offset_; }

  int outputOffset() const { return output_offset_; }

  int hiddenStateOffset() const { return hidden_state_offset_; }

  int cellStateOffset() const { return cell_state_offset_; }

private:
  int32_t current_time_ = 0;
  int32_t current_batch_ = 0;
  int32_t input_offset_ = 0;
  int32_t output_offset_ = 0;
  int32_t hidden_state_offset_ = 0;
  int32_t cell_state_offset_ = 0;

  const LstmSizeInfo &size_info_;
};

// Calculates a single LSTM gate.
// Implements the following formula:
//   gate = activate(FC(input) + FC(recurrent))
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
OMStatus
calculateLstmGate(const LstmStepManager &step_info, const GateParameters &gate_params,
                  // Input FC
                  const ActivationType *input_data, const core::OMRuntimeShape &input_weight_shape,
                  const WeightType *input_weight_data, const core::OMRuntimeShape &input_bias_shape,
                  const BiasType *input_bias_data,
                  // Recurrent FC
                  const ActivationType *recurrent_data,
                  const core::OMRuntimeShape &recurrent_weight_shape,
                  const WeightType *recurrent_weight_data, const BiasType *recurrent_bias_data,
                  // Output
                  CellType *gate_output,
                  // Scratch arrays
                  CellType *fc_output_buffer, const circle::ActivationFunctionType activation)
{
  OMStatus status = Ok;
  // Input FC
  const auto gate_output_shape = step_info.stateShape();
  {
    core::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params.input_fc_params.input_offset;
    op_params.weights_offset = gate_params.input_fc_params.weights_offset;
    op_params.output_offset = gate_params.input_fc_params.output_offset;
    op_params.output_multiplier = gate_params.input_fc_params.output_multiplier;
    op_params.output_shift = gate_params.input_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params.input_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params.input_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params.input_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params.input_fc_params.float_activation_min;

    status = FullyConnected(op_params, input_data, input_weight_shape, input_weight_data,
                            input_bias_data, gate_output_shape, gate_output);
  }

  // Recurrent FC
  {
    core::FullyConnectedParams op_params{};
    op_params.input_offset = gate_params.recurrent_fc_params.input_offset;
    op_params.weights_offset = gate_params.recurrent_fc_params.weights_offset;
    op_params.output_offset = gate_params.recurrent_fc_params.output_offset;
    op_params.output_multiplier = gate_params.recurrent_fc_params.output_multiplier;
    op_params.output_shift = gate_params.recurrent_fc_params.output_shift;
    op_params.quantized_activation_min = gate_params.recurrent_fc_params.quantized_activation_min;
    op_params.quantized_activation_max = gate_params.recurrent_fc_params.quantized_activation_max;
    op_params.float_activation_max = gate_params.recurrent_fc_params.float_activation_max;
    op_params.float_activation_min = gate_params.recurrent_fc_params.float_activation_min;

    status =
      FullyConnected(op_params, recurrent_data, recurrent_weight_shape, recurrent_weight_data,
                     recurrent_bias_data, gate_output_shape, fc_output_buffer);
  }
  {

    addElementWise<CellType>(gate_output, fc_output_buffer,
                             /*n_batch=*/gate_output_shape.dimsData()[0],
                             /*n_state=*/gate_output_shape.dimsData()[1], gate_output);

    switch (activation)
    {
        // TODO: support sigmoid
      case circle::ActivationFunctionType::ActivationFunctionType_TANH:
      {
        Tanh(gate_output_shape, input_data, gate_output_shape, gate_output);
      }
      break;
      default:
        // Only Tanh is used.
        assert(false && "Only Tanh is used");
    }
  }
  return status;
}
// Update the hidden state of the LSTM kernel using the following formula:
// updated_hidden_state = Tanh(updated_cell_state) * output_gate_output, * means
// element wise multiplication
template <typename CellType, typename ActivationType>
void updateLstmHidden(const LstmStepManager &step_info, CellType *cell_state_data_base,
                      ActivationType *hidden_state_data, const CellType &output_gate_output,
                      const core::ArithmeticParams &mul_params, int32_t cell_state_scale_power,
                      CellType *buffer)
{
  auto cell_state_shape = step_info.stateShape();
  CellType *cell_state_data = cell_state_data_base + step_info.cellStateOffset();
  // Tanh(cell_state)
  tanh(cell_state_scale_power, cell_state_shape, cell_state_data, cell_state_shape, buffer);
  // Update the hidden state
  mul(cell_state_shape, mul_params, buffer, output_gate_output,
      hidden_state_data + step_info.hiddenStateOffset());
}

// Update the cell state using the output from the forget gate, input gate, and
// cell gate Formula: updated_cell_state = forget_gate_output*cell_state +
// input_gate_output * cell_gate_output, where * denotes element wise
// multiplication
template <typename CellType>
void updateLstmCell(const LstmStepManager &step_info, CellType *cell_state_data,
                    // Gate outputs
                    CellType *forget_gate_output, const CellType *input_gate_output,
                    const CellType *cell_gate_output,
                    // Mul parameters
                    const core::ArithmeticParams &forget_cell_mul_params,
                    const core::ArithmeticParams &input_mul_params,
                    const CellStateInfo &cell_state_info, CellType *buffer)
{
  auto cell_state_shape = step_info.stateShape();
  // Forget Gate x Cell State
  Mul<CellType>(&forget_cell_mul_params, cell_state_shape.flatSize(),
                const_cast<const CellType *>(forget_gate_output),
                const_cast<const CellType *>(cell_state_data + step_info.cellStateOffset()),
                cell_state_data + step_info.cellStateOffset());
  // Input Gate x Cell Gate
  Mul<CellType>(&input_mul_params, cell_state_shape.flatSize(),
                const_cast<const CellType *>(input_gate_output),
                const_cast<const CellType *>(cell_gate_output), buffer);

  // Update the cell state
  addElementWise(cell_state_data + step_info.cellStateOffset(), buffer,
                 /*n_batch=*/cell_state_shape.dimsData()[0],
                 /*n_state=*/cell_state_shape.dimsData()[1],
                 cell_state_data + step_info.cellStateOffset());

  if (cell_state_info.cell_clip > 0)
  {
    clipping(cell_state_shape.flatSize(), cell_state_info,
             cell_state_data + step_info.cellStateOffset());
  }
}

template <typename ActivationType, typename WeightType, typename CellType, typename BiasType>
void lstmStep(const LSTMStruct &lstm_struct, const LSTMParameters &lstm_params,
              LstmStepManager &step_info, const CellStateInfo &cell_state_info,
              const ActivationType *activation_state_data, const CellType *cell_state_data,
              CellType *scratch0, CellType *scratch1, CellType *scratch2, CellType *scratch3)
{
  //   /*Step1: Calculate gate outputs to prepare cell state update*/
  CellType *gate_internal_buffer = scratch3;
  CellType *forget_gate_output = scratch0;

  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, lstm_params.forget_gate_parameters,
    // Input FC
    reinterpret_cast<const float *>(lstm_struct.input_data),
    core::OMRuntimeShape(lstm_struct.input_to_forget_weights()),
    reinterpret_cast<const float *>(lstm_struct.input_to_forget_weights_data),
    core::OMRuntimeShape(lstm_struct.forget_gate_bias()),
    reinterpret_cast<const float *>(lstm_struct.forget_gate_bias_data),
    // Recurrent FC
    reinterpret_cast<const float *>(lstm_struct.activation_state_data),
    core::OMRuntimeShape(lstm_struct.recurrent_to_forget_weights()),
    reinterpret_cast<const float *>(lstm_struct.recurrent_to_forget_weights_data), nullptr,
    // Output
    forget_gate_output, gate_internal_buffer,
    circle::ActivationFunctionType::ActivationFunctionType_TANH);
  //
  //   // Input Gate calculation;
  CellType *input_gate_output = scratch1;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, lstm_params.forget_gate_parameters,
    // Input FC
    reinterpret_cast<const float *>(lstm_struct.input_data),
    core::OMRuntimeShape(lstm_struct.input_to_input_weights()),
    reinterpret_cast<const float *>(lstm_struct.input_to_input_weights_data),
    core::OMRuntimeShape(lstm_struct.input_gate_bias()),
    reinterpret_cast<const float *>(lstm_struct.input_gate_bias_data),
    // Recurrent FC
    reinterpret_cast<const float *>(lstm_struct.activation_state_data),
    core::OMRuntimeShape(lstm_struct.recurrent_to_input_weights()),
    reinterpret_cast<const float *>(lstm_struct.recurrent_to_input_weights_data),
    /*recurrent_bias*/ nullptr,
    // Output
    input_gate_output,
    // Scratch arrays
    gate_internal_buffer, circle::ActivationFunctionType::ActivationFunctionType_TANH);

  // Cell Gate calculation
  CellType *cell_gate_output = scratch2;
  calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
    step_info, lstm_params.forget_gate_parameters,
    // Input FC
    reinterpret_cast<const float *>(lstm_struct.input_data),
    core::OMRuntimeShape(lstm_struct.input_to_cell_weights()),
    reinterpret_cast<const float *>(lstm_struct.input_to_cell_weights_data),
    core::OMRuntimeShape(lstm_struct.cell_gate_bias()),
    reinterpret_cast<const float *>(lstm_struct.cell_gate_bias_data),
    // Recurrent FC
    reinterpret_cast<const float *>(lstm_struct.activation_state_data),
    core::OMRuntimeShape(lstm_struct.recurrent_to_cell_weights()),
    reinterpret_cast<const float *>(lstm_struct.recurrent_to_cell_weights_data),
    /*recurrent_bias*/ nullptr,
    // Output
    cell_gate_output,
    // Scratch arrays
    gate_internal_buffer, circle::ActivationFunctionType::ActivationFunctionType_TANH);

  /*Step2: update the cell state */
  {
    // const InterGateParameters& inter_gate_params = op_data.inter_gate_parameters;
    CellType *updated_input_buffer = scratch1; // reuse buffer

    updateLstmCell<CellType>(step_info, const_cast<CellType *>(cell_state_data), forget_gate_output,
                             const_cast<const CellType *>(input_gate_output), cell_gate_output,
                             lstm_params.inter_gate_parameters.forget_cell_mul_params,
                             lstm_params.inter_gate_parameters.input_mul_params, cell_state_info,
                             updated_input_buffer);
  }
  //
  //   {
  //     /*Step3: update the hidden state */
  //     CellType *output_gate_output = scratch1; // reuse buffer
  //     calculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
  //       step_info, &lstm_params->output_gate_parameters,
  //       // Input FC
  //       input_data, lstm_struct->input_to_output_weights(), lstm_struct->output_gate_bias(),
  //       // Recurrent FC
  //       output_state_data, lstm_struct->recurrent_to_output_weights(), nullptr,
  //       // Output
  //       output_gate_output,
  //       // Scratch arrays
  //       gate_internal_buffer, FusedActivation::kTfLiteActSigmoid, runtime_graph);
  //     CellType *tanh_activated_cell_buffer = scratch0; // reuse buffer
  //     updateLstmHidden<CellType, ActivationType>(
  //       step_info, cell_state_data, output_state_data, output_gate_output,
  //       &lstm_params->inter_gate_parameters.output_mul_params,
  //       cell_state_info->cell_state_scale_power, tanh_activated_cell_buffer);
  //
  //     ActivationType *output_ptr = luci_interpreter::kernels::getTensorData<ActivationType>(
  //       runtime_graph->getDataByTensor(lstm_struct->output()));
  //     std::memcpy(output_ptr + step_info->outputOffset(),
  //                 output_state_data + step_info->hiddenStateOffset(),
  //                 step_info->stateShape().flatSize() * sizeof(ActivationType));
  //   }
}
} // namespace

template <typename T>
inline OMStatus
UnidirectionalSequenceLSTM(const core::OMRuntimeShape &input_shape, const float *input_data,
                           const core::OMDataType &cell_state_type, const LSTMStruct &lstm_struct,
                           const core::OMRuntimeShape &output_shape, float *output_data)
{
  assert(false && "Not Implemented Yet");
}

template <>
inline OMStatus
UnidirectionalSequenceLSTM<float>(const core::OMRuntimeShape &input_shape, const float *input_data,
                                  const core::OMDataType &cell_state_type,
                                  const LSTMStruct &lstm_struct,
                                  const core::OMRuntimeShape &output_shape, float *output_data)
{
  CellStateInfo cell_state_info{lstm_struct.options()->cell_clip(), 0, 0}; // No quantization
  LSTMParameters lstm_params{};
  prepareGateParameters<float>(lstm_params);

  const bool time_major = lstm_struct.options()->time_major();
  const auto batch_size = time_major ? input_shape.dims(1) : input_shape.dims(0);
  const auto state_dimension = output_shape.dims(1);
  const auto cell_state_type_size = core::getOMDataTypeSize(cell_state_type);

  auto scratch_0_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_1_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_2_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);
  auto scratch_3_data =
    std::make_unique<uint8_t[]>(batch_size * state_dimension * cell_state_type_size);

  // Create and fill with 0 output state tensor
  auto activation_state_data =
    std::make_unique<float[]>(core::OMRuntimeShape(lstm_struct.activation_state()).flatSize());
  std::fill_n(activation_state_data.get(),
              core::OMRuntimeShape(lstm_struct.activation_state()).flatSize(), 0);

  // Create and fill with 0 cell state tensor
  auto cell_state_data =
    std::make_unique<float[]>(core::OMRuntimeShape(lstm_struct.cell_state()).flatSize());

  std::fill_n(cell_state_data.get(), core::OMRuntimeShape(lstm_struct.cell_state()).flatSize(), 0);

  LstmSizeInfo size_info;
  size_info.time_major = time_major;
  size_info.batch_size = batch_size;
  size_info.time_steps = size_info.time_major ? input_shape.dims(0) : input_shape.dims(1);
  size_info.input_dimension = input_shape.dims(2);
  size_info.state_dimension = core::OMRuntimeShape(lstm_struct.activation_state()).dims(1);

  LstmStepManager step_info(size_info);

  // time is the first dimention, enable batch computation
  if (size_info.time_major)
  {
    for (int t = 0; t < size_info.time_steps; t++)
    {
      lstmStep<float, float, float, float>(
        lstm_struct, lstm_params, step_info, cell_state_info,
        reinterpret_cast<const float *>(lstm_struct.activation_state_data),
        reinterpret_cast<float *>(cell_state_data.get()),
        reinterpret_cast<float *>(scratch_0_data.get()),
        reinterpret_cast<float *>(scratch_1_data.get()),
        reinterpret_cast<float *>(scratch_2_data.get()),
        reinterpret_cast<float *>(scratch_3_data.get()));
      // prepare for the next time step
      step_info.updateTime();
    }
  }
  else
  {
    // batch first, unable to size the input data. single batch inference
    for (int b = 0; b < size_info.batch_size; b++)
    {
      for (int t = 0; t < size_info.time_steps; t++)
      {
        lstmStep<float, float, float, float>(
          lstm_struct, lstm_params, step_info, cell_state_info,
          reinterpret_cast<const float *>(lstm_struct.activation_state_data),
          reinterpret_cast<float *>(cell_state_data.get()),
          reinterpret_cast<float *>(scratch_0_data.get()),
          reinterpret_cast<float *>(scratch_1_data.get()),
          reinterpret_cast<float *>(scratch_2_data.get()),
          reinterpret_cast<float *>(scratch_3_data.get()));
        // prepare for the next time step
        step_info.updateTime();
      }
      // prepare for the next batch
      step_info.updateBatch();
      step_info.resetTime();
    }
  }
  return Ok;
}
} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_UNIDIRECTIONAL_SEQUENCE_LSTM_COMMON_H
