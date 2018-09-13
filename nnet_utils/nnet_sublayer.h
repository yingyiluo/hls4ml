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

#ifndef NNET_SUBLAYER_H_
#define NNET_SUBLAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct sublayer_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Number of sublayer partitions
    static const unsigned n_part = 2;
    static const unsigned i_part = 0;

    // Number of outputs; starting index (inclusive)
    static const unsigned n_sub_out = 5;
    static const unsigned i_sub_out = 0;
  
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

 template<class data_T, class res_T, typename CONFIG_T>
void compute_sublayer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_sub_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_sub_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_sub_out];

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
  
        //int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor*CONFIG_T::n_part)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor*CONFIG_T::n_part));
        int multiplier_limit = ceil(float(CONFIG_T::n_in*CONFIG_T::n_sub_out) / float(CONFIG_T::reuse_factor)); // ignoring pruning for now
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS ARRAY_RESHAPE variable=weights complete dim=1
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
    }
    

#ifdef __MTPUMPING__
    // Initialize accumulator with input biases
    int32_t Accum[CONFIG_T::n_sub_out];
    SetAccum: for(int iacc = 0; iacc < CONFIG_T::n_sub_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        int bindex = iacc+CONFIG_T::i_sub_out;
        int32_t temp = biases[bindex].V;
        Accum[iacc]  = (temp << (NBIT-IBIT));
    }

#pragma HLS array_partition variable=weights complete dim=1
#pragma HLS array_partition variable=Accum complete dim=1
 
    const bool clear = 0;
    const bool ap_clk_div2=0;
    ap_int<16> sampa1 =0;
    ap_int<16> sampa2 =0;
    ap_int<16> sampb1 =0;
    ap_int<16> sampb2 =0;
    const unsigned int roundin = (CONFIG_T::n_in >> 1 << 1);

    Product1: for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
      #pragma HLS pipeline
      // Round in the in size for multipumping
    Product2: for(int ii = 0; ii < roundin; ii+=2) {
      #pragma HLS unroll
            int weight_index = ii*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
            int weight_index2 = (ii+1)*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
            fixed_toint(sampa1, data[ii]);
            fixed_toint(sampa2, data[ii+1]);
            fixed_toint(sampb1, weights[weight_index]);
            fixed_toint(sampb2, weights[weight_index2]);
            if (sampb1 == 0 && sampb2 == 0)
              continue;
            Accum[jj] = __builtin_mac16x2(sampa1, sampa2, sampb1, sampb2, Accum[jj], clear, ap_clk_div2);
        }

      // In case odd number, calculate the rest with DSP
      Product3:if (CONFIG_T::n_in & 1) {
          fixed_toint(sampa1, data[roundin]);
          int weight_index = roundin*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
          fixed_toint(sampb1, weights[weight_index]);
          Accum[jj] = __builtin_mac16x2(sampa1, 0, sampb1, 0, Accum[jj], clear, ap_clk_div2);
        }
    }


    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_sub_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = int32_tofixed(Accum[ires]);
    }    

#else
    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor*CONFIG_T::n_part));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int weight_index = ii*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
	    int mult_index   = ii*CONFIG_T::n_sub_out+jj;
	    mult[mult_index] = cache * weights[weight_index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_sub_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
	int bias_index = iacc+CONFIG_T::i_sub_out;
        acc[iacc] = (typename CONFIG_T::accum_t) biases[bias_index];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
	    int index = ii*CONFIG_T::n_sub_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_sub_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    
#endif
}

template<class data_T, int NIN1, int NIN2>
    void merge(
        data_T data1[NIN1], 
	data_T data2[NIN2],
        data_T res[NIN1+NIN2])
{
    for(int ii=0; ii<NIN1; ii++){
        res[ii] = data1[ii];
    }
    for(int ii=0; ii<NIN2; ii++){
        res[NIN1+ii] = data2[ii];
    }
}




}//end namespace
  
  
#endif
