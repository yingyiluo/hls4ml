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
    // partitioning arrays cyclically to go with roll factors?
};
 
 
 template<class data_T, class res_T, typename CONFIG_T>
   void compute_layer(
		      data_T    data[CONFIG_T::n_in],
		      res_T     res[CONFIG_T::n_out],
		      typename CONFIG_T::weight_t  weights[CONFIG_T::n_in][CONFIG_T::n_out],
		      typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
 {
   data_T cache;
   typename CONFIG_T::accum_t mult[CONFIG_T::n_in][CONFIG_T::n_out];
   typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

   // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
#pragma HLS function_instantiate variable=weights,biases

   if (CONFIG_T::io_type == io_parallel){
     // For parallel inputs:
     //   - completely partition arrays -- target fabric
     //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE
        #pragma HLS ARRAY_PARTITION variable=weights complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
     if (CONFIG_T::reuse_factor > 1) {
       int multiplier_limit  = ceil(CONFIG_T::n_in*CONFIG_T::n_out / CONFIG_T::reuse_factor);
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
     }
   } else if (CONFIG_T::io_type == io_serial){
     // TODO: Fill out the directives for serial input
     // #pragma HLS ALLOCATION instances=mul limit=1 operation
   }

   // Do the matrix-multiply
 Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
     cache = data[ii];
   Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
       mult[ii][jj] = cache * weights[ii][jj];
     }
   }

   // Initialize accumulator with input biases
 ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
     acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
   }

   // Accumulate multiplication result
 Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
   Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
       acc[jj] += mult[ii][jj];
     }
   }

   // Cast to "res_t" type 
 Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
     res[ires] = (res_T) (acc[ires]);
   }
 } 


 template<class data_T, class res_T, typename CONFIG_T>
   void compute_layer2(
		      data_T    data[CONFIG_T::n_in],
		      res_T     res[CONFIG_T::n_out],
		      typename CONFIG_T::weight_t  weights[CONFIG_T::n_in][CONFIG_T::n_out],
		      typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
 {
   data_T cache;
   typename CONFIG_T::accum_t mult[CONFIG_T::n_in][CONFIG_T::n_out];
   typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    ap_fixed<18,8> weights2[32][1] = {-0.0582222938538, 0, -0.437459200621, -0.176295682788, 0.0648417845368, 0.187429219484, 0.156936720014, -0.357166618109, -0.169738873839, -0.324651628733, 0.237157344818, 0.0639939531684, -0.386845320463, -0.00481491070241, -0.159990355372, 0.0899929478765, 0.125663176179, 0.0247628651559, -0.427373498678, -0.0069064674899, 0.0494951717556, -0.0939490869641, -0.238442763686, -0.292703747749, -0.322070419788, -0.36728823185, 0.0726772025228, 0.155764341354, 0.117759428918, -0.10209748894, -0.181185156107, 0.0284108463675};
    bool useWeights2=false;
    if(CONFIG_T::n_in==32 && CONFIG_T::n_out==1) useWeights2=true;

   // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
#pragma HLS function_instantiate variable=weights,biases
    #pragma HLS function_instantiate variable=weights2

   if (CONFIG_T::io_type == io_parallel){
     // For parallel inputs:
     //   - completely partition arrays -- target fabric
     //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE
     #pragma HLS ARRAY_PARTITION variable=weights complete
     #pragma HLS ARRAY_PARTITION variable=weights2 complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
     if (CONFIG_T::reuse_factor > 1) {
       int multiplier_limit  = ceil(CONFIG_T::n_in*CONFIG_T::n_out / CONFIG_T::reuse_factor);
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
     }
   } else if (CONFIG_T::io_type == io_serial){
     // TODO: Fill out the directives for serial input 
     // #pragma HLS ALLOCATION instances=mul limit=1 operation
   }

   if(useWeights2){                                                                                                                                                   
     std::cout << "USE WEIGHTS2 HACK" << std::endl;

      Product1_ww: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
     cache = data[ii];
     Product2_ww: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
     mult[ii][jj] = cache * weights[ii][jj];
      }
      }

     //   Product1_w2: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
     // cache = data[ii];
     // mult[ii][0] = cache * weights2[ii][0];
     // }

   }                                                                                                                                                                  
   else{                                                                                                                                                              
   Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {                                                                                                             
       cache = data[ii];                                                                                                                                              
     Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {                                                                                                          
	 mult[ii][jj] = cache * weights[ii][jj];                                                                                                                      
       }                                                                                                                                                              
     }                                                                                                                                                                
   }
   // Initialize accumulator with input biases
 ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
     acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
   }

   // Accumulate multiplication result
 Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
   Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
       acc[jj] += mult[ii][jj];
     }
   }

   // Cast to "res_t" type
 Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
     res[ires] = (res_T) (acc[ires]);
   }
 } 



 template<class data_T, class res_T, typename CONFIG_T>
   void compute_layer3(
		      data_T    data[CONFIG_T::n_in],
		      res_T     res[CONFIG_T::n_out],
		      ap_fixed<18,8>  weights[32][1],
		      typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
 {
   data_T cache;
   typename CONFIG_T::accum_t mult[CONFIG_T::n_in][CONFIG_T::n_out];
   typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

   // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
#pragma HLS function_instantiate variable=weights,biases

   if (CONFIG_T::io_type == io_parallel){
     // For parallel inputs:
     //   - completely partition arrays -- target fabric
     //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE
        #pragma HLS ARRAY_PARTITION variable=weights complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
     if (CONFIG_T::reuse_factor > 1) {
       int multiplier_limit  = ceil(CONFIG_T::n_in*CONFIG_T::n_out / CONFIG_T::reuse_factor);
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
     }
   } else if (CONFIG_T::io_type == io_serial){
     // TODO: Fill out the directives for serial input
     // #pragma HLS ALLOCATION instances=mul limit=1 operation
   }

   // Do the matrix-multiply
 Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
     cache = data[ii];
   Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
       mult[ii][jj] = cache * weights[ii][jj];
     }
   }

   // Initialize accumulator with input biases
 ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
     acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
   }

   // Accumulate multiplication result
 Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
   Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
       acc[jj] += mult[ii][jj];
     }
   }

   // Cast to "res_t" type 
 Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
     res[ires] = (res_T) (acc[ires]);
   }
 } 




}

#endif
