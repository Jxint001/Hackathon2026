#include "graph.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace npu {
namespace {
using Json = nlohmann::json;

// The DOM callback parser repeatedly scans growing arrays at object_end.
// A linear SAX validation pass keeps duplicate-key rejection without that
// quadratic overhead on large graph edge arrays.
struct KeyValidator : nlohmann::json_sax<Json> {
    std::vector<std::unordered_set<std::string>> keys;
    bool null() override { return true; }
    bool boolean(bool) override { return true; }
    bool number_integer(number_integer_t) override { return true; }
    bool number_unsigned(number_unsigned_t) override { return true; }
    bool number_float(number_float_t, const string_t&) override { return true; }
    bool string(string_t&) override { return true; }
    bool binary(binary_t&) override { return true; }
    bool start_object(std::size_t) override { keys.emplace_back(); return true; }
    bool end_object() override { keys.pop_back(); return true; }
    bool key(string_t& name) override {
        if (!keys.back().insert(name).second) throw std::invalid_argument("duplicate JSON object key");
        return true;
    }
    bool start_array(std::size_t) override { return true; }
    bool end_array() override { return true; }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& error) override {
        throw std::invalid_argument(error.what());
    }
};

Id read_id(const Json& value) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
        throw std::invalid_argument("node IDs must be nonnegative integers");
    return value.get<Id>();
}

Cost read_cost(const Json& value, bool placeholder = false) {
    if (!value.is_number_integer()) throw std::invalid_argument("expected an integer size or cycle count");
    if (value.is_number_unsigned() && value.get<Id>() > static_cast<Id>(kInfinity))
        throw std::invalid_argument("size or cycle count exceeds the supported integer range");
    const Cost result = value.get<Cost>();
    if (result < 0 && !(placeholder && result == -1))
        throw std::invalid_argument("negative size or compute cycle count");
    if (result > kInfinity) throw std::invalid_argument("cycle count too large");
    return result;
}

void check_dag(const std::vector<std::vector<int>>& successors) {
    std::vector<int> degree(successors.size(), 0);
    for (const auto& list : successors) for (int v : list) ++degree[v];
    std::vector<int> queue;
    queue.reserve(successors.size());
    for (std::size_t i = 0; i < degree.size(); ++i)
        if (!degree[i]) queue.push_back(static_cast<int>(i));
    for (std::size_t i = 0; i < queue.size(); ++i)
        for (int v : successors[queue[i]]) if (--degree[v] == 0) queue.push_back(v);
    if (queue.size() != successors.size()) throw std::invalid_argument("graph contains a dependency cycle");
}
}

Graph read_graph(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.u8string());
    KeyValidator validator;
    if (!Json::sax_parse(input, &validator)) throw std::invalid_argument("invalid graph JSON");
    input.clear(); input.seekg(0);
    const Json root = Json::parse(input);
    if (!root.is_object()) throw std::invalid_argument("graph must be a JSON object");
    for (const char* field : {"tensors", "ops", "edges"})
        if (!root.contains(field) || !root.at(field).is_array())
            throw std::invalid_argument(std::string("missing array: ") + field);

    Graph g;
    for (const auto& entry : root.at("tensors")) {
        Tensor t;
        t.id = read_id(entry.at("id"));
        const auto position = entry.at("pos").get<std::string>();
        if (position == "DDR") t.storage = -1;
        else if (position == "L1") t.storage = 0;
        else if (position == "UB") t.storage = 1;
        else throw std::invalid_argument("unknown tensor position: " + position);
        t.bytes = read_cost(entry.at("size"));
        g.max_id = std::max(g.max_id, t.id);
        g.tensors.push_back(std::move(t));
    }
    for (const auto& entry : root.at("ops")) {
        Operation op;
        op.id = read_id(entry.at("id"));
        op.name = entry.at("op").get<std::string>();
        op.copy = op.name == "COPY_IN" ? 1 : op.name == "COPY_OUT" ? 2 : 0;
        const auto pipe = entry.at("pipe").get<std::string>();
        if (pipe == "PIPE_MTE2") op.pipe = 0;
        else if (pipe == "PIPE_MTE3") op.pipe = 1;
        else if (pipe == "PIPE_M") op.pipe = 2;
        else if (pipe == "PIPE_V") op.pipe = 3;
        else if (pipe == "PIPE_FIX") op.pipe = 4;
        else throw std::invalid_argument("unsupported pipe: " + pipe);
        op.cycles = read_cost(entry.at("cycles"), op.copy != 0);
        g.max_id = std::max(g.max_id, op.id);
        g.ops.push_back(std::move(op));
    }
    std::sort(g.tensors.begin(), g.tensors.end(), [](const Tensor& a, const Tensor& b) { return a.id < b.id; });
    std::sort(g.ops.begin(), g.ops.end(), [](const Operation& a, const Operation& b) { return a.id < b.id; });
    if (g.ops.size() + g.tensors.size() > static_cast<std::size_t>(INT32_MAX))
        throw std::invalid_argument("too many nodes");
    struct Node { bool op; int index; };
    std::unordered_map<Id, Node> ids;
    ids.reserve(g.ops.size() + g.tensors.size());
    for (int i = 0; i < static_cast<int>(g.ops.size()); ++i) {
        if (!ids.emplace(g.ops[i].id, Node{true, i}).second)
            throw std::invalid_argument("duplicate node ID");
        if (!g.ops[i].copy) g.work.push_back(i);
    }
    for (int i = 0; i < static_cast<int>(g.tensors.size()); ++i)
        if (!ids.emplace(g.tensors[i].id, Node{false, i}).second)
            throw std::invalid_argument("duplicate node ID");

    std::set<std::pair<Id, Id>> seen_edges;
    for (const auto& entry : root.at("edges")) {
        const Id src = read_id(entry.at("source")), dst = read_id(entry.at("target"));
        if (!seen_edges.emplace(src, dst).second) throw std::invalid_argument("duplicate edge");
        const auto a = ids.find(src), b = ids.find(dst);
        if (a == ids.end() || b == ids.end()) throw std::invalid_argument("edge endpoint does not exist");
        const auto x = a->second, y = b->second;
        if (x.op && y.op) {
            g.direct.emplace_back(x.index, y.index);
            g.direct_bytes.push_back(entry.contains("data_size") ? read_cost(entry.at("data_size")) : 0);
        }
        else if (x.op) {
            g.ops[x.index].outputs.push_back(y.index);
            g.tensors[y.index].producers.push_back(x.index);
        } else if (y.op) {
            g.ops[y.index].inputs.push_back(x.index);
            g.tensors[x.index].consumers.push_back(y.index);
        } else throw std::invalid_argument("tensor-to-tensor edges are not supported");
    }
    std::vector<std::vector<int>> full(g.ops.size());
    g.pred.resize(g.ops.size());
    g.succ.resize(g.ops.size());
    const auto add = [&g, &full](int a, int b) {
        full[a].push_back(b);
        if (!g.ops[a].copy && !g.ops[b].copy) {
            g.succ[a].push_back(b);
            g.pred[b].push_back(a);
        }
    };
    for (auto& t : g.tensors) {
        unique_sort(t.producers); unique_sort(t.consumers);
        for (int a : t.producers) for (int b : t.consumers) add(a, b);
    }
    for (const auto& edge : g.direct) add(edge.first, edge.second);
    for (auto& op : g.ops) { unique_sort(op.inputs); unique_sort(op.outputs); }
    for (auto& list : full) unique_sort(list);
    for (auto& list : g.pred) unique_sort(list);
    for (auto& list : g.succ) unique_sort(list);
    check_dag(full);
    g.topo = dependency_order(g, false);
    return g;
}

std::vector<int> dependency_order(const Graph& g, bool depth_first) {
    std::vector<int> depth(g.ops.size()), degree(g.ops.size());
    std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
    for (int op : g.work) {
        degree[op] = static_cast<int>(g.pred[op].size());
        if (!degree[op]) ready.push(op);
    }
    std::vector<int> result;
    while (!ready.empty()) {
        const int op = ready.top(); ready.pop(); result.push_back(op);
        for (int v : g.succ[op]) {
            depth[v] = std::max(depth[v], depth[op] + 1);
            if (--degree[v] == 0) ready.push(v);
        }
    }
    if (result.size() != g.work.size()) throw std::invalid_argument("compute dependency cycle");
    if (!depth_first) return result;
    const auto less = [&depth](int a, int b) {
        if (depth[a] != depth[b]) return depth[a] < depth[b];
        return a > b;
    };
    std::vector<int> stack;
    for (int op : g.work) if (g.succ[op].empty()) stack.push_back(op);
    std::sort(stack.begin(), stack.end(), less);
    std::vector<char> done(g.ops.size());
    result.clear();
    while (!stack.empty()) {
        const int op = stack.back();
        if (done[op]) { stack.pop_back(); continue; }
        std::vector<int> predecessors;
        for (int p : g.pred[op]) if (!done[p]) predecessors.push_back(p);
        if (predecessors.empty()) { result.push_back(op); done[op] = 1; stack.pop_back(); }
        else {
            std::sort(predecessors.begin(), predecessors.end(), less);
            stack.insert(stack.end(), predecessors.begin(), predecessors.end());
        }
    }
    return result;
}

bool valid_plan(const Graph& g, const Plan& plan) {
    if (plan.orders.size() < 2 || plan.orders.size() > 5) return false;
    std::vector<int> owner(g.ops.size(), -1);
    std::vector<std::vector<int>> next = g.succ;
    std::size_t total = 0;
    for (int core = 0; core < static_cast<int>(plan.orders.size()); ++core) {
        const auto& order = plan.orders[core];
        int last = -1;
        for (int op : order) {
            if (op < 0 || op >= static_cast<int>(g.ops.size()) || g.ops[op].copy || owner[op] >= 0) return false;
            owner[op] = core; ++total;
            if (last != -1) next[last].push_back(op);
            last = op;
        }
    }
    if (total != g.work.size()) return false;
    for (const auto& tensor : g.tensors) {
        int writer_core = -1;
        for (int op : tensor.producers) if (!g.ops[op].copy) {
            if (writer_core >= 0 && owner[op] != writer_core) return false;
            writer_core = owner[op];
        }
    }
    for (auto& list : next) unique_sort(list);
    try { check_dag(next); } catch (const std::invalid_argument&) { return false; }
    return true;
}

void write_plan(const Graph& g, const Plan& plan, const std::filesystem::path& output) {
    Json mapping = Json::object(), schedules = Json::array();
    for (int op : g.work) mapping[std::to_string(g.ops[op].id)] = op;
    for (const auto& order : plan.orders) schedules.push_back(order);
    const Json result{{"node_to_subgraph", std::move(mapping)}, {"core_schedules", std::move(schedules)}};
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    auto temporary = output;
    temporary += ".npu_sched.tmp";
    for (std::uint64_t suffix = 1; std::filesystem::exists(temporary) || std::filesystem::is_symlink(temporary); ++suffix) {
        temporary = output;
        temporary += ".npu_sched.tmp." + std::to_string(suffix);
    }
    try {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot create output file");
        out << result.dump(2) << '\n';
        out.flush();
        if (!out) throw std::runtime_error("failed to write the complete plan");
        out.close();
        if (!out) throw std::runtime_error("failed to close output file");
        std::error_code ec;
        std::filesystem::rename(temporary, output, ec);
        if (ec) {
            // Windows does not replace an existing destination with rename.
            if (std::filesystem::is_regular_file(output)) std::filesystem::remove(output);
            std::filesystem::rename(temporary, output);
        }
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        throw;
    }
}
} // namespace npu
