#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <CL/cl.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sstream>
#include <iomanip>

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    try {
        // read input data
        std::ifstream fin("input.txt");
        if (!fin) {
            std::cerr << "Couldn't open input.txt, aborting\n";
            return 1;
        }

        int n;
        fin >> n;

        std::vector<float> a(n);
        for (int i = 0; i < n; ++i) {
            fin >> a[i];
        }

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                    cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            program.build(devices);
        }
        catch (cl::Error const & e)
        {         
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * n);
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        // load named kernels from opencl source
        const int block_size = 256;
        cl::Kernel forward_block_scan_iteration(program, "forward_block_scan_iteration");
        cl::Kernel backward_block_sum_propagation_iteration(program, "backward_block_sum_propagation_iteration");

        // forward loop
        int lvl;
        for (lvl = 1; lvl < n; lvl *= block_size) {
            queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n, &a[0]);

            forward_block_scan_iteration.setArg(0, n);
            forward_block_scan_iteration.setArg(1, lvl);
            forward_block_scan_iteration.setArg(2, dev_input);
            forward_block_scan_iteration.setArg(3, dev_output);
            forward_block_scan_iteration.setArg(4, cl::__local(sizeof(float) * block_size));
            forward_block_scan_iteration.setArg(5, cl::__local(sizeof(float) * block_size));

            int eff_n = (n - 1) / lvl + 1;
            cl::NDRange eff_global_range(block_size * ((eff_n - 1) / block_size + 1));
            cl::NDRange local_range(block_size);
            queue.enqueueNDRangeKernel(forward_block_scan_iteration, cl::NullRange, eff_global_range, local_range);

            queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n, &a[0]);
        }

        // backward loop
        for (lvl /= block_size; lvl > 1; lvl /= block_size) {
            queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * n, &a[0]);

            backward_block_sum_propagation_iteration.setArg(0, n);
            backward_block_sum_propagation_iteration.setArg(1, lvl);
            backward_block_sum_propagation_iteration.setArg(2, dev_input);
            backward_block_sum_propagation_iteration.setArg(3, dev_output);

            int eff_n = (n - 1) / (lvl / block_size) + 1;
            cl::NDRange eff_global_range(block_size * ((eff_n - 1) / block_size + 1));
            cl::NDRange local_range(block_size);
            queue.enqueueNDRangeKernel(backward_block_sum_propagation_iteration, 
                                       cl::NullRange, 
                                       eff_global_range, 
                                       local_range);

            queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * n, &a[0]);
        }

        // output the result
        std::ofstream fout("output.txt");
        fout << std::fixed << std::setprecision(3);
        for (int i = 0; i < n; ++i) {
            fout << a[i] << " ";
        }
        fout << '\n';
    }
    catch (cl::Error const & e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
