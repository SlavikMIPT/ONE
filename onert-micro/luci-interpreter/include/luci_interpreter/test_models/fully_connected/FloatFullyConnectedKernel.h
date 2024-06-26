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

#ifndef LUCI_INTERPRETER_TEST_MODELS_FULLY_CONNECTED_KERNEL_FLOAT_H
#define LUCI_INTERPRETER_TEST_MODELS_FULLY_CONNECTED_KERNEL_FLOAT_H

#include "TestDataFullyConnectedBase.h"

namespace luci_interpreter
{
namespace test_kernel
{
namespace fully_connected_float
{

/*
 * FullyConnected Kernel:
 *
 * Input(1, 16)   Weight(4, 16)   Bias(4)
 *            \        |         /
 *             \       |        /
 *               FullyConnected
 *                     |
 *                Output(1, 4)
 */

const unsigned char test_kernel_model_circle[] = {
  0x18, 0x00, 0x00, 0x00, 0x43, 0x49, 0x52, 0x30, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x08, 0x00, 0x10, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x60, 0x01, 0x00, 0x00, 0xa8, 0x02, 0x00, 0x00, 0xc4, 0x02, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x4c, 0x01, 0x00, 0x00, 0x44, 0x01, 0x00, 0x00, 0x3c, 0x01, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xe2, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x40, 0xc0, 0x00, 0x00, 0x80, 0x40,
  0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0xc0,
  0x00, 0x00, 0x80, 0xc0, 0x00, 0x00, 0xa0, 0xc0, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0xc0,
  0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x40, 0x40,
  0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0xc1, 0x00, 0x00, 0xc0, 0xc0, 0x00, 0x00, 0xe0, 0x40,
  0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0xc0,
  0x00, 0x00, 0x80, 0xc0, 0x00, 0x00, 0xa0, 0xc0, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0xc0,
  0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x40, 0x40,
  0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0xc1, 0x00, 0x00, 0xc0, 0xc0, 0x00, 0x00, 0xe0, 0x40,
  0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0xc0,
  0x00, 0x00, 0x80, 0xc0, 0x00, 0x00, 0xa0, 0xc0, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0xc0,
  0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x40, 0x40,
  0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0xc1, 0x00, 0x00, 0xc0, 0xc0, 0x00, 0x00, 0xe0, 0x40,
  0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0xc0,
  0x00, 0x00, 0x80, 0xc0, 0x00, 0x00, 0xa0, 0xc0, 0x00, 0x00, 0xc0, 0x40, 0x00, 0x00, 0xe0, 0xc0,
  0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x40, 0x40,
  0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0xc1, 0x00, 0x00, 0xc0, 0xc0, 0x00, 0x00, 0xe0, 0x40,
  0x00, 0x00, 0xa0, 0x40, 0x8c, 0xff, 0xff, 0xff, 0x90, 0xff, 0xff, 0xff, 0x94, 0xff, 0xff, 0xff,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x14, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c, 0x00,
  0x07, 0x00, 0x08, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x90, 0xff, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x6f, 0x75, 0x74, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0xb4, 0xff, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x62, 0x69, 0x61, 0x73, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xd8, 0xff, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x11, 0x00, 0x00, 0x00,
  0x4f, 0x4e, 0x45, 0x2d, 0x74, 0x66, 0x6c, 0x69, 0x74, 0x65, 0x32, 0x63, 0x69, 0x72, 0x63, 0x6c,
  0x65, 0x00, 0x00, 0x00};

const std::vector<float> input_data = {
  17.491695, 15.660671, 4.7347794,  -15.796822, 20.4776,    18.438372, -0.7529831, 10.671711,
  10.699566, 3.1682281, -22.776001, 1.527811,   -0.1198349, -5.748741, -5.1772327, 20.06879};

const std::vector<float> reference_output_data = {263.84323, 260.84323, 259.84323, 266.84323};

} // namespace fully_connected_float

class TestDataFloatFullyConnected : public TestDataFullyConnectedBase<float>
{
public:
  TestDataFloatFullyConnected()
  {
    _input_data = fully_connected_float::input_data;
    _reference_output_data = fully_connected_float::reference_output_data;
    _test_kernel_model_circle = fully_connected_float::test_kernel_model_circle;
  }

  ~TestDataFloatFullyConnected() override = default;
};

} // namespace test_kernel
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_TEST_MODELS_FULLY_CONNECTED_KERNEL_FLOAT_H
