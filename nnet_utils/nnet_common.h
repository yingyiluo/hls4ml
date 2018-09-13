//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include "ap_fixed.h"

#ifdef __MTPUMPING__
#include "dsp_builtins.h"
#pragma SDS data access_pattern(a:SEQUENTIAL, b:SEQUENTIAL)
#include <bitset>
#define MBIT 32
#define NBIT 16
#define IBIT 4

inline ap_fixed<NBIT, IBIT> int32_tofixed(int32_t inp) {
  ap_int<MBIT> shiftbit (inp >> (NBIT-IBIT));
  ap_fixed<NBIT, IBIT> fixed_result(0);

#ifdef __OVERFLOW__
  bool debug=false;
  // Number of bit won't be interpreted
  static const unsigned int L = (MBIT - (NBIT-IBIT)) - NBIT;
  // Testing overflow bits
  static const unsigned int testvalue = ((1<<L) - 1 ) << (NBIT-1);
  static const unsigned int maxint = (((1<<(IBIT-1)) - 1)<< (NBIT-IBIT)) ; // Keep one for sign
  static const unsigned int fixsignb = 1<<(NBIT-1);
  static const unsigned int getdecimal = (1<<(NBIT-IBIT))-1;

  bool isneg =inp & (1<<(MBIT-1));

  if (debug)
  {
    const int b = 32 -(NBIT-IBIT);
    std::cout << "input " << std::bitset<32>(inp) << std::endl;
    std::cout << "shifted  " << std::bitset<b>(shiftbit) << std::endl;
    std::cout << "testvalue" << std::bitset<b>(testvalue) << std::endl;
    std::cout << "maxvalue " << std::bitset<b>(maxint) << std::endl;
    std::cout << "size bit " << std::bitset<b>(1<<(NBIT-1)) << std::endl;
    std::cout << "decimal  " << std::bitset<b>(getdecimal) << std::endl;
  }

  bool isover = false;
  if (isneg)
  {
    //assuming twos complement notation 
    isover = (~(shiftbit-1)) & testvalue;
  }
  else isover = shiftbit & testvalue;

  //std::cout << "isneg " << isneg <<" isover " << isover << std::endl;
  if (isover)
  {
    int decimalbit = shiftbit & getdecimal;
    if (isneg)
      shiftbit = (1 << (NBIT-1)) | decimalbit;
    else
      shiftbit = maxint | decimalbit;

    if(debug)
    {
      ap_fixed<NBIT, IBIT> temp;
      temp.range(NBIT-1, 0) = shiftbit;
      std::cout << "decimal bit  " << std::bitset<NBIT>(decimalbit) <<"  value " << temp<< std::endl;
    }
  }
#endif

  //Obtain the last bit
  fixed_result.range(NBIT-1, 0) = shiftbit;
  return fixed_result;
}

inline void fixed_toint(ap_int<NBIT> &to, ap_fixed<NBIT, IBIT> from) {
  to = from.V;
  //to = from.range(NBIT-1, 0);
}
#endif

namespace nnet {

// Common type definitions
enum io_type {io_parallel = 0, io_serial};

// Default data types (??) TODO: Deprecate
typedef ap_fixed<16,4>  weight_t_def;
typedef ap_fixed<16,4>  bias_t_def;
typedef ap_fixed<32,10> accum_t_def;

}

#endif
