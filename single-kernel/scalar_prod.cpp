#include "common.h"

#include <iostream>
#include <type_traits>
#include <iomanip>

//using namespace cl::sycl;
namespace s = cl::sycl;

enum class invocation_method {
  ndrange,
  hierarchical,
  scoped
};

template<typename T, invocation_method>
class ScalarProdKernel;
template<typename T, invocation_method>
class ScalarProdKernelHierarchical;
template<typename T, invocation_method>
class ScalarProdKernelScoped;

template<typename T, invocation_method>
class ScalarProdReduction;
template<typename T, invocation_method>
class ScalarProdReductionHierarchical;
template<typename T, invocation_method>
class ScalarProdReductionScoped;

class ScalarProdGatherKernel;


template<typename T, 
  invocation_method invocation_kind = invocation_method::ndrange>
class ScalarProdBench
{
protected:    
    std::vector<T> input1;
    std::vector<T> input2;
    std::vector<T> output;
    BenchmarkArgs args;

    PrefetchedBuffer<T, 1> input1_buf;
    PrefetchedBuffer<T, 1> input2_buf;
    PrefetchedBuffer<T, 1> output_buf;

public:
  ScalarProdBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast<T>(1);
      input2[i] = static_cast<T>(2);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    
    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the hostbuffer must first be copied to device
      auto intermediate_product = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      if(invocation_kind == invocation_method::ndrange){
        cl::sycl::nd_range<1> ndrange (args.problem_size, args.local_size);

        cgh.parallel_for<class ScalarProdKernel<T, invocation_kind>>(ndrange,
          [=](cl::sycl::nd_item<1> item) 
          {
            size_t gid= item.get_global_linear_id();
            intermediate_product[gid] = in1[gid] * in2[gid];
          });
      }
      else if (invocation_kind == invocation_method::hierarchical){
        cgh.parallel_for_work_group<class ScalarProdKernelHierarchical<T, invocation_kind>>(
          cl::sycl::range<1>{args.problem_size / args.local_size},
          cl::sycl::range<1>{args.local_size},
          [=](cl::sycl::group<1> grp){
            grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
              size_t gid = idx.get_global_id(0);
              intermediate_product[gid] = in1[gid] * in2[gid];
            });
          });
      }
      else if (invocation_kind == invocation_method::scoped){
#ifdef __HIPSYCL__
        cgh.parallel<class ScalarProdKernelScoped<T, invocation_kind>>(
          cl::sycl::range<1>{args.problem_size / args.local_size},
          cl::sycl::range<1>{args.local_size},
          [=](cl::sycl::group<1> grp, cl::sycl::physical_item<1>){
            grp.distribute_for([&](cl::sycl::sub_group, cl::sycl::logical_item<1> idx){
              size_t gid = idx.get_global_id(0);
              intermediate_product[gid] = in1[gid] * in2[gid];
            });
          });
#endif
      }
    }));

    // std::cout << "Multiplication of vectors completed" << std::endl;

    auto array_size = args.problem_size;
    auto wgroup_size = args.local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while (array_size!= 1) {
      auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

      events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);
      
          // local memory for reduction
          auto local_mem = s::accessor <T, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh};
          cl::sycl::nd_range<1> ndrange (n_wgroups*wgroup_size, wgroup_size);
    
          if(invocation_kind == invocation_method::ndrange) {
            cgh.parallel_for<class ScalarProdReduction<T, invocation_kind>>(ndrange,
            [=](cl::sycl::nd_item<1> item) 
              {
                size_t gid= item.get_global_linear_id();
                size_t lid = item.get_local_linear_id();

                // initialize local memory to 0
                local_mem[lid] = 0; 

                for(int i = 0; i < elements_per_thread; ++i) {
                  int input_element = gid + i * n_wgroups * wgroup_size;
                  
                  if(input_element < array_size)
                    local_mem[lid] += global_mem[input_element];
                }

                item.barrier(s::access::fence_space::local_space);

                for(size_t stride = wgroup_size/elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  if(lid < stride) {
                    for(int i = 0; i < elements_per_thread-1; ++i){
                      local_mem[lid] += local_mem[lid + stride + i];
                    }
                  }
                  item.barrier(s::access::fence_space::local_space);
                }
                
                // Only one work-item per work group writes to global memory 
                if (lid == 0) {
                  global_mem[item.get_global_id()] = local_mem[0];
                }
              });
          }
          else if(invocation_kind == invocation_method::hierarchical){
            cgh.parallel_for_work_group<class ScalarProdReductionHierarchical<T, invocation_kind>>(
              cl::sycl::range<1>{n_wgroups}, cl::sycl::range<1>{wgroup_size},
              [=](cl::sycl::group<1> grp){
                
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t gid = idx.get_global_id(0);
                  const size_t lid = idx.get_local_id(0);

                  // initialize local memory to 0
                  local_mem[lid] = 0; 

                  for(int i = 0; i < elements_per_thread; ++i) {
                    int input_element = gid + i * n_wgroups * wgroup_size;
                  
                    if(input_element < array_size)
                      local_mem[lid] += global_mem[input_element];
                  }
                });

                for(size_t stride = wgroup_size/elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  
                    const size_t lid = idx.get_local_id(0);
                    
                    if(lid < stride) {
                      for(int i = 0; i < elements_per_thread-1; ++i){
                        local_mem[lid] += local_mem[lid + stride + i];
                      }
                    }
                  });
                }
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t lid = idx.get_local_id(0);
                  if(lid == 0)
                    global_mem[grp.get_id(0) * grp.get_local_range(0)] = local_mem[0];
                });
              });
          } else if(invocation_kind == invocation_method::scoped) {
#ifdef __HIPSYCL__
            cgh.parallel<class ScalarProdReductionScoped<T, invocation_kind>>(
              cl::sycl::range<1>{n_wgroups}, cl::sycl::range<1>{wgroup_size},
              [=](cl::sycl::group<1> grp, cl::sycl::physical_item<1>){
                grp.distribute_for([&](cl::sycl::sub_group, cl::sycl::logical_item<1> idx){
                  const size_t gid = idx.get_global_id(0);
                  const size_t lid = idx.get_local_id(0);

                  // initialize local memory to 0
                  local_mem[lid] = 0; 

                  for(int i = 0; i < elements_per_thread; ++i) {
                    int input_element = gid + i * n_wgroups * wgroup_size;
                  
                    if(input_element < array_size)
                      local_mem[lid] += global_mem[input_element];
                  }
                });
                for(size_t stride = wgroup_size/elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  grp.distribute_for([&](cl::sycl::sub_group, cl::sycl::logical_item<1> idx){
                  
                    const size_t lid = idx.get_local_id(0);
                    
                    if(lid < stride) {
                      for(int i = 0; i < elements_per_thread-1; ++i){
                        local_mem[lid] += local_mem[lid + stride + i];
                      }
                    }
                  });
                }
                grp.single_item([&](){
                  global_mem[grp.get_id(0) * grp.get_local_range(0)] = local_mem[0];
                });
              });
#endif
          }
        }));
      
      events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);
      
          cgh.parallel_for<ScalarProdGatherKernel>(cl::sycl::range<1>{n_wgroups},
                                                   [=](cl::sycl::id<1> idx){
            global_mem[idx] = global_mem[idx * wgroup_size];
          });
        }));
      array_size = n_wgroups;
    }
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    auto expected = static_cast <T>(0);

    auto output_acc = output_buf.template get_access<s::access::mode::read>();

    for(size_t i = 0; i < args.problem_size; i++) {
        expected += input1[i] * input2[i];
    }

    //std::cout << "Scalar product on CPU =" << expected << std::endl;
    //std::cout << "Scalar product on Device =" << output[0] << std::endl;

    // Todo: update to type-specific test (Template specialization?)
    const auto tolerance = 0.00001f;
    if(std::fabs(expected - output_acc[0]) > tolerance) {
      pass = false;
    }

    return pass;
  }
  
  static std::string getBenchmarkName() {

    std::string variant;
    if(invocation_kind == invocation_method::ndrange){
      variant = "NDRange_";
    }
    else if(invocation_kind == invocation_method::hierarchical) {
      variant = "Hierarchical_";
    }
    else if(invocation_kind == invocation_method::scoped) {
      variant = "Scoped_";
    }

    std::stringstream name;
    name << "ScalarProduct_" << variant;
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  if(app.shouldRunNDRangeKernels()) {
    app.run<ScalarProdBench<int, invocation_method::ndrange>>();
    app.run<ScalarProdBench<long long, invocation_method::ndrange>>();
    app.run<ScalarProdBench<float, invocation_method::ndrange>>();
    app.run<ScalarProdBench<double, invocation_method::ndrange>>();
  }

  app.run<ScalarProdBench<int, invocation_method::hierarchical>>();
  app.run<ScalarProdBench<long long, invocation_method::hierarchical>>();
  app.run<ScalarProdBench<float, invocation_method::hierarchical>>();
  app.run<ScalarProdBench<double, invocation_method::hierarchical>>();

#ifdef __HIPSYCL__
  app.run<ScalarProdBench<int, invocation_method::scoped>>();
  app.run<ScalarProdBench<long long, invocation_method::scoped>>();
  app.run<ScalarProdBench<float, invocation_method::scoped>>();
  app.run<ScalarProdBench<double, invocation_method::scoped>>();
#endif

  return 0;
}
