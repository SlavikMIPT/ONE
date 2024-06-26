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

#ifndef LUCI_INTERPRETER_TEST_MODELS_FLOAT_UNIDIRECTIONAL_LSTM_KERNEL_H
#define LUCI_INTERPRETER_TEST_MODELS_FLOAT_UNIDIRECTIONAL_LSTM_KERNEL_H

#include "TestDataUnidirectionalLSTMBase.h"

namespace luci_interpreter
{
namespace test_kernel
{
namespace unidir_lstm_float
{
/*
 * UnidirectionalLSTM Kernel:
 *
 *      Input(1, 4, 4)
 *            |
 *      UnidirectionalLSTM
 *            |
 *      Output(1, 4, 2)
 */
const unsigned char test_kernel_model_circle[] = {
  0x18, 0x00, 0x00, 0x00, 0x43, 0x49, 0x52, 0x30, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x08, 0x00, 0x10, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0xb4, 0x01, 0x00, 0x00, 0xd0, 0x05, 0x00, 0x00, 0xec, 0x05, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0xa0, 0x01, 0x00, 0x00, 0x94, 0x01, 0x00, 0x00, 0x8c, 0x01, 0x00, 0x00, 0x84, 0x01, 0x00, 0x00,
  0x64, 0x01, 0x00, 0x00, 0x3c, 0x01, 0x00, 0x00, 0x1c, 0x01, 0x00, 0x00, 0xfc, 0x00, 0x00, 0x00,
  0xe4, 0x00, 0x00, 0x00, 0xcc, 0x00, 0x00, 0x00, 0x9c, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xa0, 0xfe, 0xff, 0xff,
  0xd2, 0xfe, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x84, 0x9b, 0x3c, 0xbe,
  0x48, 0xfc, 0x22, 0xbf, 0x35, 0x43, 0xba, 0x3e, 0x18, 0x5c, 0xdb, 0x3e, 0x4b, 0x05, 0xdd, 0xbe,
  0xf5, 0x0f, 0x1e, 0xbf, 0x1f, 0x2e, 0x09, 0x3f, 0x9e, 0xb5, 0x2f, 0x3f, 0xfe, 0xfe, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0xca, 0xfe, 0x05, 0x3f, 0xea, 0x91, 0x06, 0xbe,
  0x77, 0xf4, 0xa7, 0xbe, 0x3f, 0x02, 0x23, 0xbf, 0xd1, 0xdc, 0x94, 0xbd, 0xc2, 0xdd, 0xb1, 0xbe,
  0x45, 0x13, 0xc8, 0x3e, 0x7f, 0x6b, 0xef, 0x3e, 0x2a, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0xec, 0xde, 0xe2, 0xbe, 0xf6, 0x46, 0x01, 0xbf, 0xed, 0x4d, 0x97, 0xbd,
  0xf2, 0xed, 0x09, 0xbf, 0x88, 0x4c, 0xe1, 0x3e, 0x60, 0x74, 0x89, 0x3e, 0x29, 0x79, 0x75, 0x3c,
  0x9b, 0x8f, 0xdb, 0xbe, 0x56, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0xe7, 0xc8, 0x6a, 0x3e, 0x16, 0x06, 0x8b, 0xbd, 0x49, 0xf5, 0xe5, 0x3e, 0x01, 0xfb, 0xf0, 0x3e,
  0x7c, 0x48, 0x10, 0xbf, 0x12, 0xd8, 0x94, 0xbe, 0x9a, 0xec, 0xaf, 0x3e, 0x4c, 0x1a, 0xdb, 0xbe,
  0x82, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  0x00, 0x00, 0x80, 0x3f, 0x96, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x8d, 0xd4, 0xdb, 0xbd, 0xfe, 0x63, 0xd1, 0x3e, 0x9f, 0x60, 0x1a, 0x3d,
  0xa1, 0x48, 0x0b, 0xbf, 0xc6, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x3a, 0xb2, 0xca, 0x3e, 0x58, 0x1a, 0x04, 0xbf, 0xe6, 0x76, 0x9f, 0x3e, 0x61, 0xa7, 0xd8, 0x3e,
  0xe2, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xf1, 0x48, 0x5c, 0xbe,
  0xcd, 0x54, 0xad, 0x3c, 0x9f, 0x8e, 0xbf, 0x3e, 0x69, 0xac, 0xfd, 0x3d, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x93, 0x70, 0x21, 0xbf, 0x76, 0x27, 0x8e, 0x3c, 0x97, 0xe3, 0xc5, 0x3e, 0xe5, 0x7d, 0x8c, 0x3e,
  0xf4, 0xff, 0xff, 0xff, 0xf8, 0xff, 0xff, 0xff, 0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x04, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0xc4, 0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00,
  0xcc, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x08, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47,
  0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00,
  0x0b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x00, 0x04,
  0x01, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x06, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0xe4, 0x02, 0x00, 0x00, 0x98, 0x02, 0x00, 0x00, 0x54, 0x02, 0x00, 0x00, 0x20, 0x02, 0x00, 0x00,
  0xec, 0x01, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x88, 0x01, 0x00, 0x00, 0x58, 0x01, 0x00, 0x00,
  0x24, 0x01, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0xbc, 0x00, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x60, 0xfd, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x53, 0x74, 0x61, 0x74,
  0x65, 0x66, 0x75, 0x6c, 0x50, 0x61, 0x72, 0x74, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x65, 0x64, 0x43,
  0x61, 0x6c, 0x6c, 0x3a, 0x30, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xec, 0xfd, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x0c, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x6c, 0x73, 0x74, 0x6d, 0x2f,
  0x7a, 0x65, 0x72, 0x6f, 0x73, 0x31, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xdc, 0xfd, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x39, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x0c, 0xfe, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x38, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x3c, 0xfe, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x37, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x6c, 0xfe, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x36, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x9c, 0xfe, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f,
  0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x35, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xc8, 0xfe, 0xff, 0xff, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61,
  0x6e, 0x74, 0x34, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0xf4, 0xfe, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x33, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x24, 0xff, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x32, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x54, 0xff, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x31, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x84, 0xff, 0xff, 0xff,
  0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x61, 0x72, 0x69, 0x74, 0x68, 0x2e, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x14, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x00, 0x00, 0x07, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x2f, 0x6c,
  0x73, 0x74, 0x6d, 0x2f, 0x7a, 0x65, 0x72, 0x6f, 0x73, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f,
  0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x5f, 0x31, 0x3a,
  0x30, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x0c, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x2c, 0x11, 0x00, 0x00, 0x00, 0x4f, 0x4e, 0x45, 0x2d, 0x74, 0x66, 0x6c, 0x69,
  0x74, 0x65, 0x32, 0x63, 0x69, 0x72, 0x63, 0x6c, 0x65, 0x00, 0x00, 0x00};

const std::vector<float> input_data = {
  -3.509163, -18.256927, 6.4799614, 10.296598, 30.371328, 18.692572, 10.12867,   -26.44944,
  25.324795, 3.8303719,  20.93112,  22.603086, -4.308655, 2.3276749, -5.9565907, 25.611776};

const std::vector<float> reference_output_data = {0.7613201,      -0.7570043,   0.0480366,
                                                  -4.3364323e-11, -0.7613433,   1.3437739e-08,
                                                  -0.7613537,     -7.000451e-08};

} // namespace unidir_lstm_float

class TestDataFloatUnidirectionalLSTM : public TestDataUnidirectionalLSTMBase<float>
{
public:
  TestDataFloatUnidirectionalLSTM()
  {
    _input_data = unidir_lstm_float::input_data;
    _reference_output_data = unidir_lstm_float::reference_output_data;
    _test_kernel_model_circle = unidir_lstm_float::test_kernel_model_circle;
  }

  ~TestDataFloatUnidirectionalLSTM() override = default;
};

} // namespace test_kernel
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_TEST_MODELS_FLOAT_UNIDIRECTIONAL_LSTM_KERNEL_H
