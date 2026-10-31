// 程序入口：解析命令行、读取计算图、调用调度器并写出方案，按提交契约返回退出码。
#include "json_parser.hpp"
#include "scheduler.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <string>

namespace {
// 表示命令行用法错误，继承标准异常以携带错误消息。
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// 表示目录创建或方案写出失败。
struct OutputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// 解析后的程序调用参数。
struct Options {
    std::filesystem::path input, output;
    int cores = 4;
    bool version = false;
};
// 解析命令行、应用默认参数并禁止覆盖输入图。
Options parse_options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version") {
            o.version = true;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            auto split = arg.find('=');
            auto key = arg.substr(0, split);
            std::string value;
            if (key != "-n" && key != "--cores" && key != "-o" && key != "--output") {
                throw UsageError("unknown option: " + key);
            }
            if (split != std::string::npos) {
                value = arg.substr(split + 1);
            } else {
                if (i + 1 == argc || argv[i + 1][0] == '-') {
                    throw UsageError("missing value: " + key);
                }
                value = argv[++i];
            }
            if (value.empty()) {
                throw UsageError("empty option value: " + key);
            }
            if (key == "-n" || key == "--cores") {
                if (value.size() != 1 || value[0] < '2' || value[0] > '5') {
                    throw UsageError("cores must be 2, 3, 4 or 5");
                }
                o.cores = value[0] - '0';
            } else {
                o.output = std::filesystem::u8path(value);
            }
        } else {
            if (!o.input.empty()) {
                throw UsageError("exactly one input graph is required");
            }
            o.input = std::filesystem::u8path(arg);
        }
    }
    if (o.version) {
        return o;
    }
    if (o.input.empty()) {
        throw UsageError("usage: npu_sched <graph.json> -n <2..5> -o <plan.json>");
    }
    if (o.output.empty()) {
        o.output = o.input.parent_path() / (o.input.stem().u8string() + "_multicore_res.json");
    }
    std::error_code ec;
    if (std::filesystem::equivalent(o.input, o.output, ec) ||
        std::filesystem::absolute(o.input).lexically_normal() ==
            std::filesystem::absolute(o.output).lexically_normal()) {
        throw UsageError("output must not overwrite the input graph");
    }
    return o;
}
// 读取并解析输入图。
npu::Problem load_problem(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw npu::InputError("cannot open input graph");
    }
    std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw npu::InputError("cannot read input graph");
    }
    auto json = npu::JsonParser(data).parse();
    return npu::read_graph(json);
}

// 写入并关闭文件，确认数据完整写出。
void write_plan(const std::filesystem::path &path, const std::string &text) {
    try {
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw OutputError("cannot create output file");
        }
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        output.close();
        if (!output) {
            throw OutputError("cannot write output file");
        }
    } catch (const std::filesystem::filesystem_error &e) {
        throw OutputError(e.what());
    }
}
}
// 执行图读取、求解、方案序列化与写盘。
int main(int argc, char **argv) {
    std::locale::global(std::locale::classic());
    try {
        Options args = parse_options(argc, argv);
        if (args.version) {
            std::cout << "npu_sched v9 (9.0.0)\n";
            return 0;
        }
        auto problem = load_problem(args.input);
        auto solution = npu::solve(problem, args.cores);
        write_plan(args.output, npu::serialize_plan(problem, solution, args.cores));
        return 0;
    } catch (const UsageError &e) {
        std::cerr << e.what() << '\n';
        return 2;
    } catch (const npu::InputError &e) {
        std::cerr << e.what() << '\n';
        return 3;
    } catch (const OutputError &e) {
        std::cerr << e.what() << '\n';
        return 4;
    } catch (const npu::NoPlan &e) {
        std::cerr << e.what() << '\n';
        return 5;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
