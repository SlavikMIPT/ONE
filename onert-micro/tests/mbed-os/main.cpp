/*
* Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mbed.h"
#undef ARG_MAX
#define LUCI_LOG 0
#include <luci_interpreter/Interpreter.h>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "modelfully.circle.h"

// Blinking rate in milliseconds
#define BLINKING_RATE 500ms

luci_interpreter::Interpreter interpreter(circle_model_raw, true);
float neural_sin(float x)
{
  auto input_data = reinterpret_cast<float *>(interpreter.allocateInputTensor(0));
  *input_data = x;
  interpreter.interpret();
  auto data = interpreter.readOutputTensor(0);
  return *reinterpret_cast<float *>(data);
}
void print_float(float x)
{
  int tmp = x * 1000 - static_cast<int>(x) * 1000;
  std::cout << (tmp >= 0 ? "" : "-") << static_cast<int>(x) << ".";
  int zeros_to_add = 0;
  for (int i = 100; i >= 1; i = i / 10)
  {
    if (tmp / i != 0)
      break;
    zeros_to_add++;
  }
  for (int i = 0; i < zeros_to_add; ++i)
  {
    std::cout << "0";
  }
  std::cout << (tmp >= 0? tmp : -tmp) << "\n";
}
int main()
{
#ifdef LED1
  DigitalOut led(LED1);
#else
  bool led;
#endif
  for (int i = 1; i < 10; ++i)
  {
    float res = neural_sin(M_PI / i);
    std::cout << "NEURAL SIN ";
    print_float(res);
    std::cout << "ACTUAL SIN ";
    print_float(std::sin(M_PI / i));
  }

  while (true)
  {
    ThisThread::sleep_for(BLINKING_RATE);
  }
}
