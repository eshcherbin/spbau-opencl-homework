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
    std::vector<cl::Kernel> kernels;

    try {
        // read input data
        std::ifstream fin("input.txt");
        if (!fin) {
            std::cerr << "Couldn't open input.txt, aborting\n";
            return 1;
        }

        int n, m;
        fin >> n >> m;

        std::vector<float> a(n * n);
        std::vector<float> b(m * m);
        std::vector<float> c(n * n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fin >> a[i * n + j];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                fin >> b[i * m + j];
            }
        }

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution_2d.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                    cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            std::stringstream m_ss;
            m_ss << "-DM=" << m;
            program.build(devices, m_ss.str().c_str());
        }
        catch (cl::Error const & e)
        {         
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * n * n);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * m * m);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

        // copy from CPU to GPU
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "convolution_2d");
        kernel.setArg(0, dev_a);
        kernel.setArg(1, n);
        kernel.setArg(2, dev_b);
        kernel.setArg(3, m);
        kernel.setArg(4, dev_c);
        int block_size = 16;
        cl::NDRange global_range(block_size * ((n - 1) / block_size + 1), block_size * ((n - 1) / block_size + 1));
        cl::NDRange local_range(block_size, block_size);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(int) * n * n, &c[0]);

        // output the result
        std::ofstream fout("output.txt");
        fout << std::fixed << std::setprecision(3);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fout << c[i * n + j] << ' ';
            }
            fout << '\n';
        }
    }
    catch (cl::Error const & e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
