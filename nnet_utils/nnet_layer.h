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

#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>


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

struct layer_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete

        int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }
    

#ifdef __MTPUMPING__
    // Initialize accumulator with input biases
    int32_t Accum[CONFIG_T::n_out];
    SetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        Accum[iacc] = 0;
    }

//#pragma HLS array_partition variable=data complete dim=1
#pragma HLS array_partition variable=weights complete dim=1
#pragma HLS array_partition variable=Accum complete dim=1
 
    bool clear = 0;
    bool ap_clk_div2=0;
    ap_int<16> sampa1 =0;
    ap_int<16> sampa2 =0;
    ap_int<16> sampb1 =0;
    ap_int<16> sampb2 =0;

    Product1: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
      #pragma HLS pipeline
      // Round in the in size for multipumping
      unsigned int roundin = (CONFIG_T::n_in >> 1 << 1);
    Product2: for(int ii = 0; ii < roundin; ii+=2) {
      #pragma HLS unroll
            fixed_toint(sampa1, data[ii]);
            fixed_toint(sampa2, data[ii+1]);
            fixed_toint(sampb1, weights[ii*CONFIG_T::n_out+jj]);
            fixed_toint(sampb2, weights[(ii+1)*CONFIG_T::n_out+jj]);
            if (sampb1 == 0 && sampb2 == 0)
              continue;
            Accum[jj] = __builtin_mac16x2(sampa1, sampa2, sampb1, sampb2, Accum[jj], clear, ap_clk_div2);
        }

      // In case odd number, calculate the rest with DSP
      Product3:if (CONFIG_T::n_in & 1) {
          fixed_toint(sampa1, data[roundin]);
          fixed_toint(sampb1, weights[roundin*CONFIG_T::n_out+jj]);
          Accum[jj] = __builtin_mac16x2(sampa1, 0, sampb1, 0, Accum[jj], clear, ap_clk_div2);
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    Accum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[iacc] += int32_tofixed(Accum[iacc]);
    }


    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    

#else
    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        //acc[iacc] = 0;
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    
#endif
}

}

#endif
