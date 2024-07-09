/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizeWeightsOnly.h"
#include "QuantizationUtils.h"
#include "OMHuffmanTranscoder.h"

#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>

#include <cmath>
#include <vector>
#include <functional>
#include <limits>
#include <iostream>
using namespace luci;

namespace
{

using IterFunc = std::function<void(uint32_t *, loco::TensorShape &, int32_t)>;

void iterate_per_channel(CircleConst *node, int32_t &channel_dim_index, IterFunc func)
{
  loco::TensorShape dimension;
  dimension.rank(4);
  uint32_t indices[4] = {
    0,
  };

  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    assert(false);
    return;
  }

  for (indices[0] = 0; indices[0] < dimension.dim(0).value(); indices[0]++)
  {
    for (indices[1] = 0; indices[1] < dimension.dim(1).value(); indices[1]++)
    {
      for (indices[2] = 0; indices[2] < dimension.dim(2).value(); indices[2]++)
      {
        for (indices[3] = 0; indices[3] < dimension.dim(3).value(); indices[3]++)
        {
          func(indices, dimension, channel_dim_index);
        }
      }
    }
  }
}
//#include <iostream>
// TODO Reduce duplicate code with QuantizeDequantizeWeights
template <loco::DataType out_type>
void sym_wquant_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                            std::vector<float> &scaling_factor, std::vector<float> &nudged_min,
                            std::vector<float> &nudged_max, int32_t &channel_dim_index)
{
  assert(node->dtype() == loco::DataType::FLOAT32);
  assert(out_type == loco::DataType::S4 || out_type == loco::DataType::U8 ||
         out_type == loco::DataType::S16);

  const int32_t kMaxScale = max_for_sym_quant(out_type);
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::U8>();
  std::cout << size << " SIZE\n";


//  node->dtype(out_type);      // change the type of tensor

//  node->size<out_type>(size); // resize tensor
  std::cout <<"--------------------------------- HEREEEEEEEEEEEEEEE\n";
  std::vector<uint8_t> input;

  for (uint32_t i = 0; i < size; ++i)
  {
    input.push_back(node->at<out_type>(i));
    node->at<out_type>(i) = 0;
  }
}

void cal_minmax_per_channel(CircleConst *node, std::vector<float> &min, std::vector<float> &max,
                            int32_t &channel_dim_index)
{
  loco::TensorShape dimension;
  dimension.rank(4);

  if (!get_channel_dim_index(node, dimension, channel_dim_index))
  {
    throw std::runtime_error("Failed to find channel index in " + node->name());
  }
  auto size = dimension.dim(channel_dim_index).value();

  std::vector<bool> has_min_max_value(size, false);
  min.resize(size);
  max.resize(size);

  auto cal_minmax = [&](uint32_t *indices, loco::TensorShape &dimension, int channel_dim_index) {
    int channel_idx = indices[channel_dim_index];
    auto data = node->at<loco::DataType::FLOAT32>(cal_offset(dimension, indices));
    if (has_min_max_value[channel_idx])
    {
      min[channel_idx] = data < min[channel_idx] ? data : min[channel_idx];
      max[channel_idx] = data > max[channel_idx] ? data : max[channel_idx];
    }
    else
    {
      min[channel_idx] = data;
      max[channel_idx] = data;
      has_min_max_value[channel_idx] = true;
    }
  };

  iterate_per_channel(node, channel_dim_index, cal_minmax);
}

} // namespace

namespace luci
{

void QuantizeWeightsOnly::quantize_weights(luci::CircleConst *weights)
{
  // Find min/max per channel-wise
//  std::cout <<"--------------------------------- HEREEEEEEEEEEEEEEE\n";

  if (granularity == QuantizationGranularity::ChannelWise)
  {
    uint32_t size = weights->size<loco::DataType::U8>();
//    std::cout <<"--------------------------------- HEREEEEEEEEEEEEEEE\n";
    onert_micro::core::HuffmanTranscoder<uint8_t> transcoder;
    std::vector<uint8_t> input;

    for (uint32_t i = 0; i < size; ++i)
    {
      input.push_back(weights->at<loco::DataType::U8>(i));
    }
    std::vector<uint8_t> encoded = transcoder.encodeInputArray(input);
    auto decoded = transcoder.decodeEncodedArray(encoded);
    if(decoded == input)
      std::cout << "EQUAL\n";
    else
    {
      std::cout << "NOT EQUAL!!!!!!!!!!!!!!\n";
      std::cout << decoded.size() << " decoded.size()\n";
      std::cout << input.size() << " input.size()\n";

    }
    weights->size<loco::DataType::U8>(decoded.size());
    for (uint32_t i = 0; i < size; ++i)
    {
      weights->at<loco::DataType::U8>(i) = decoded[i];
    }
    static size_t input_size_sum = 0, encoded_size_sum = 0;
    input_size_sum += input.size();
    encoded_size_sum += encoded.size();
    std::cout  << (int)((100 - (float)encoded_size_sum / input_size_sum * 100) + 0.5) << "% compression\n";

//    auto quantparam = weights->quantparam();
//    if (quantparam == nullptr)
//    {
      // Find min/max on the fly
      // NOTE This is for the case when QuantizeDequantizeWeights is skipped
      // TODO Reduce duplicate codes
//      std::vector<float> min;
//      std::vector<float> max;
//      int32_t channel_dim_index = 0;
//
//      cal_minmax_per_channel(weights, min, max, channel_dim_index);
//
//      std::vector<float> nudged_min(min.size());
//      std::vector<float> nudged_max(min.size());
//      std::vector<float> scaling_factor(min.size());
//      std::vector<int64_t> zp(min.size());
//
//      if (output_type == loco::DataType::S4)
//      {
//        sym_wquant_per_channel<loco::DataType::S4>(weights, min, max, scaling_factor, nudged_min,
//                                                   nudged_max, channel_dim_index);
//      }
//      else if (output_type == loco::DataType::U8)
//      {
//        std::cout <<"--------------------------------- HEREEEEEEEEEEEEEEE\n";
//        sym_wquant_per_channel<loco::DataType::U8>(weights, min, max, scaling_factor, nudged_min,
//                                                   nudged_max, channel_dim_index);
//      }
//      else if (output_type == loco::DataType::S16)
//      {
//        sym_wquant_per_channel<loco::DataType::S16>(weights, min, max, scaling_factor, nudged_min,
//                                                    nudged_max, channel_dim_index);
//      }
//      else
//      {
//        throw std::runtime_error("Weights-only quantization supports s8 and s16");
//      }

//      auto quantparam = std::make_unique<CircleQuantParam>();
//      quantparam->scale = scaling_factor;
//      quantparam->zerop = zp;
//      quantparam->quantized_dimension = channel_dim_index;
//      weights->quantparam(std::move(quantparam));

//      return;
//    }
  }
  else
    throw std::runtime_error("Weights-only quantization does not support layer-wise");
}

void QuantizeWeightsOnly::visit(luci::CircleConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visits node: " << node->name() << std::endl;
//  std::cout <<"--------------------------------- HEREEEEEEEEEEEEEEE\n";

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
//  if (!is_quantized(weights))
//  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
//  }
}

void QuantizeWeightsOnly::visit(luci::CircleFullyConnected *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visit node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
//  if (!is_quantized(weights))
//  {
    auto new_weights = luci::clone(weights);
    node->weights(new_weights);
    quantize_weights(new_weights);
//  }
}

void QuantizeWeightsOnly::visit(luci::CircleDepthwiseConv2D *node)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsOnly visits node: " << node->name() << std::endl;

  auto weights = loco::must_cast<luci::CircleConst *>(node->filter());
//  if (!is_quantized(weights))
//  {
    auto new_weights = luci::clone(weights);
    node->filter(new_weights);
    quantize_weights(new_weights);
//  }
}

void QuantizeWeightsOnly::visit(luci::CircleNode *) {}

} // namespace luci
