#include "graph.hpp"
#include "strategy.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
struct Options {
    std::filesystem::path input, output;
    int cores = 4;
    bool version = false;
};

Options parse(int argc, char** argv) {
    Options result;
    bool input_seen = false, cores_seen = false, output_seen = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--version" && argc == 2) {
            result.version = true;
            return result;
        }
        if (!arg.empty() && arg.front() == '-') {
            const auto eq = arg.find('=');
            const std::string key = arg.substr(0, eq);
            const bool is_cores = key == "-n" || key == "--cores";
            const bool is_output = key == "-o" || key == "--output";
            if (!is_cores && !is_output) throw std::invalid_argument("unknown option: " + key);
            std::string value;
            if (eq != std::string::npos) value = arg.substr(eq + 1);
            else {
                if (++i == argc) throw std::invalid_argument("missing value for " + key);
                value = argv[i];
                if (!value.empty() && value.front() == '-')
                    throw std::invalid_argument("missing value for " + key);
            }
            if (value.empty()) throw std::invalid_argument("empty value for " + key);
            if (is_cores) {
                if (cores_seen || value.size() != 1 || value[0] < '2' || value[0] > '5')
                    throw std::invalid_argument("cores must be one of 2, 3, 4, 5, specified once");
                cores_seen = true;
                result.cores = value[0] - '0';
            } else {
                if (output_seen) throw std::invalid_argument("output specified more than once");
                output_seen = true;
                result.output = std::filesystem::u8path(value);
            }
        } else {
            if (input_seen || arg.empty()) throw std::invalid_argument("expected exactly one graph path");
            input_seen = true;
            result.input = std::filesystem::u8path(arg);
        }
    }
    if (!input_seen) throw std::invalid_argument("usage: npu_sched <graph.json> -n <2..5> -o <plan.json>");
    if (!output_seen)
        result.output = result.input.parent_path() /
            (result.input.stem().u8string() + "_multicore_res.json");
    if (std::filesystem::absolute(result.input).lexically_normal() ==
        std::filesystem::absolute(result.output).lexically_normal())
        throw std::invalid_argument("input and output must be different files");
    std::error_code ec;
    if (std::filesystem::equivalent(result.input, result.output, ec) && !ec)
        throw std::invalid_argument("input and output refer to the same file");
    return result;
}
}

int main(int argc, char** argv) {
    Options options;
    try { options = parse(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    if (options.version) { std::cout << "npu_sched 1.0.0\n"; return 0; }

    npu::Graph graph;
    try { graph = npu::read_graph(options.input); }
    catch (const std::bad_alloc&) { std::cerr << "insufficient memory\n"; return 1; }
    catch (const std::exception& e) { std::cerr << "input error: " << e.what() << '\n'; return 3; }

    npu::Plan plan;
    try {
        plan = npu::solve(graph, options.cores);
        if (!npu::valid_plan(graph, plan)) {
            std::cerr << "unable to construct a valid schedule\n";
            return 5;
        }
    } catch (const std::exception& e) {
        std::cerr << "solver error: " << e.what() << '\n';
        return 1;
    }
    try { npu::write_plan(graph, plan, options.output); }
    catch (const std::exception& e) { std::cerr << "output error: " << e.what() << '\n'; return 4; }
    return 0;
}
