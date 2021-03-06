/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2017-2020 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/NEON/functions/NETransposeConvLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/UtilsEx.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{

NETransposeConvLayer::NETransposeConvLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _conv_f(),
      _upsample_f(),
      _flip_weights(),
      _scaled_output(),
      _weights_flipped(),
      _flip_axis(),
      _original_weights(nullptr),
      _input(nullptr),
      _info(),
      _is_prepared(false)
{
}

Status NETransposeConvLayer::validate(const ITensorInfo *input, const ITensorInfo *weights,
                                      const ITensorInfo *bias, const ITensorInfo *output,
                                      const PadStrideInfo &info, unsigned int invalid_right,
                                      unsigned int invalid_bottom)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16,
                                                       DataType::QASYMM8, DataType::QASYMM8_SIGNED);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, input);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(weights, input);
  const unsigned int width_idx =
      get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
  const unsigned int height_idx =
      get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::HEIGHT);
  ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) != weights->dimension(height_idx));
  ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(width_idx) < 1);

  auto out_dims = transposeconv_output_dimensions(
      input->dimension(width_idx), input->dimension(height_idx), weights->dimension(width_idx),
      weights->dimension(height_idx), info, invalid_right, invalid_bottom);

  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
  if (bias != nullptr)
  {
    if (is_data_type_quantized_asymmetric(input->data_type()))
    {
      ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
    }
    else
    {
      ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
    }
  }

  if (output->tensor_shape().total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    const TensorShape output_shape = compute_transposeconv_output_shape(out_dims, *input, *weights);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimX) != output_shape.x(),
                                    "Output's width is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimY) != output_shape.y(),
                                    "Output's height is invalid.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(Window::DimZ) != output_shape.z(),
                                    "Output's depth is invalid.");
  }

  unsigned int pad_left = 0;
  unsigned int pad_right = 0;
  unsigned int pad_top = 0;
  unsigned int pad_bottom = 0;
  const TensorShape scale_out_shape = compute_transposeconv_upsampled_shape(
      *input, *weights, info, out_dims, invalid_right, invalid_bottom, pad_left, pad_right, pad_top,
      pad_bottom);
  TensorInfo scale_out_info(
      input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(scale_out_shape));
  const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

  const unsigned int batches_idx =
      get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::BATCHES);
  const unsigned int channel_idx =
      get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::CHANNEL);
  ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(batches_idx) !=
                              scale_out_info.dimension(batches_idx));
  ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(channel_idx) !=
                              scale_out_info.dimension(channel_idx));

  ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayer::validate(&scale_out_info, weights, bias, output,
                                                           conv_info, WeightsInfo()));

  return Status{};
}

void NETransposeConvLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias,
                                     ITensor *output, const PadStrideInfo &info,
                                     unsigned int invalid_right, unsigned int invalid_bottom)
{
  // Perform validation step
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_ERROR_THROW_ON(NETransposeConvLayer::validate(
      input->info(), weights->info(), (bias == nullptr) ? nullptr : bias->info(), output->info(),
      info, invalid_right, invalid_bottom));

  const DataLayout data_layout = input->info()->data_layout();
  const unsigned int width_idx =
      get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const unsigned int height_idx =
      get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
  auto out_dims = transposeconv_output_dimensions(
      input->info()->dimension(width_idx), input->info()->dimension(height_idx),
      weights->info()->dimension(width_idx), weights->info()->dimension(height_idx), info,
      invalid_right, invalid_bottom);

  const TensorShape output_shape =
      compute_transposeconv_output_shape(out_dims, *input->info(), *weights->info());

  _input = input;
  _original_weights = weights;
  _info = info;
  _is_prepared = false;

  unsigned int pad_left = 0;
  unsigned int pad_right = 0;
  unsigned int pad_top = 0;
  unsigned int pad_bottom = 0;
  const unsigned int stride_x = info.stride().first;
  const unsigned int stride_y = info.stride().second;

  // Output auto initialization if not yet initialized
  auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(),
                     input->info()->quantization_info());

  _flip_axis.allocator()->init(TensorInfo(TensorShape(2U), 1, DataType::U32));
  _memory_group.manage(&_scaled_output);

  _weights_flipped.allocator()->init(weights->info()->clone()->set_data_layout(data_layout));
  _flip_weights.configure(weights, &_weights_flipped, &_flip_axis);

  // setup the function to convolve the upscaled output
  const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

  const TensorShape scale_out_shape = compute_transposeconv_upsampled_shape(
      *input->info(), *weights->info(), info, out_dims, invalid_right, invalid_bottom, pad_left,
      pad_right, pad_top, pad_bottom);

  const PadStrideInfo upsample_info(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom,
                                    DimensionRoundingType::FLOOR);

  TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(),
                            input->info()->quantization_info());
  scale_out_info.set_data_layout(data_layout);
  _scaled_output.allocator()->init(scale_out_info);

  _upsample_f.configure(input, &_scaled_output, upsample_info);

  _conv_f.configure(&_scaled_output, &_weights_flipped, bias, output, conv_info);

  // Setup flip axis data
  _flip_axis.allocator()->allocate();
  auto axis_data = reinterpret_cast<uint32_t *>(_flip_axis.buffer());
  axis_data[0] = static_cast<uint32_t>(width_idx);
  axis_data[1] = static_cast<uint32_t>(height_idx);

  _scaled_output.allocator()->allocate();
}

void NETransposeConvLayer::run()
{
  prepare();

  MemoryGroupResourceScope scope_mg(_memory_group);

  _upsample_f.run();
  _conv_f.run();
}

void NETransposeConvLayer::prepare()
{
  if (!_is_prepared)
  {
    ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

    // Run weights flipping and mark original weights tensor as unused
    _weights_flipped.allocator()->allocate();
    _flip_weights.run();
    _original_weights->mark_as_unused();

    // Prepare convolution
    _conv_f.prepare();

    _is_prepared = true;
  }
}
} // namespace arm_compute
