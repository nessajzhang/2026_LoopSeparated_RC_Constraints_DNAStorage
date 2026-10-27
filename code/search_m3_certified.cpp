// Bounded exact native producer for arbitrary 64-bit DNA m=3 window sets.
// All discards use guarded integer arithmetic.  The standalone Python checker
// reconstructs both split children, every SCC/vector bound, and root coverage.
// This file does not assert that the threshold can be reached within a pilot.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using U64 = std::uint64_t;
using U128 = unsigned __int128;
constexpr U64 FULL = std::numeric_limits<U64>::max();
constexpr int N = 16;
constexpr unsigned EXPONENT = 4;
constexpr unsigned STEPS = 8;
volatile std::sig_atomic_t stopped = 0;
void signal_stop(int) { stopped = 1; }

U64 checked(U128 value) {
    if (value > std::numeric_limits<U64>::max()) throw std::overflow_error("uint64 arithmetic overflow");
    return static_cast<U64>(value);
}
U128 power128(U64 value, unsigned power) {
    U128 result = 1;
    const U128 maximum = ~static_cast<U128>(0);
    for (unsigned i = 0; i < power; ++i) {
        if (value && result > maximum / value) throw std::overflow_error("uint128 power overflow");
        result *= value;
    }
    return result;
}
U128 multiply128(U128 a, U128 b) {
    const U128 maximum = ~static_cast<U128>(0);
    if (a && b > maximum / a) throw std::overflow_error("uint128 comparison overflow");
    return a * b;
}
struct Rational {
    U64 numerator = 0, denominator = 1;
    Rational() = default;
    Rational(U64 n, U64 d) {
        if (!d) throw std::invalid_argument("zero rational denominator");
        U64 g = std::gcd(n, d);
        numerator = n / g; denominator = d / g;
    }
};
bool less(const Rational& a, const Rational& b) {
    return static_cast<U128>(a.numerator) * b.denominator < static_cast<U128>(b.numerator) * a.denominator;
}
Rational multiply(Rational a, Rational b) {
    U64 g1 = std::gcd(a.numerator, b.denominator), g2 = std::gcd(b.numerator, a.denominator);
    return Rational(checked(static_cast<U128>(a.numerator / g1) * (b.numerator / g2)),
                    checked(static_cast<U128>(a.denominator / g2) * (b.denominator / g1)));
}
bool below_target(const Rational& value, unsigned exponent, U64 target_n, U64 target_d) {
    return multiply128(value.numerator, power128(target_d, exponent)) <=
           multiply128(value.denominator, power128(target_n, exponent));
}
using Matrix = std::array<std::array<unsigned, N>, N>;
Matrix matrix(U64 mask) {
    Matrix result{};
    for (unsigned edge = 0; edge < 64; ++edge) if ((mask >> edge) & 1ULL) result[edge / 4][edge % 16] += 1;
    return result;
}
struct Block { std::vector<int> vertices; std::vector<U64> vector; };
struct Bound { Rational value; std::vector<Block> blocks; };
Bound collatz(const Matrix& mat) {
    std::array<unsigned, N> reach{};
    for (int i = 0; i < N; ++i) {
        reach[i] = 1U << i;
        for (int j = 0; j < N; ++j) if (mat[i][j]) reach[i] |= 1U << j;
    }
    for (int k = 0; k < N; ++k) for (int i = 0; i < N; ++i) if (reach[i] & (1U << k)) reach[i] |= reach[k];
    unsigned assigned = 0;
    Bound result;
    for (int start = 0; start < N; ++start) {
        if (assigned & (1U << start)) continue;
        Block block;
        for (int j = 0; j < N; ++j) if ((reach[start] & (1U << j)) && (reach[j] & (1U << start))) {
            block.vertices.push_back(j); assigned |= 1U << j;
        }
        if (block.vertices.size() == 1 && !mat[start][start]) continue;
        block.vector.assign(block.vertices.size(), 1);
        auto apply = [&](const std::vector<U64>& x) {
            std::vector<U64> y(x.size(), 0);
            for (std::size_t i = 0; i < x.size(); ++i) {
                U128 sum = 0;
                for (std::size_t j = 0; j < x.size(); ++j) sum += static_cast<U128>(mat[block.vertices[i]][block.vertices[j]]) * x[j];
                y[i] = checked(sum);
            }
            return y;
        };
        for (unsigned step = 0; step < STEPS; ++step) {
            auto y = apply(block.vector);
            for (std::size_t i = 0; i < y.size(); ++i) block.vector[i] = checked(static_cast<U128>(block.vector[i]) + y[i]);
        }
        auto y = block.vector;
        for (unsigned p = 0; p < EXPONENT; ++p) y = apply(y);
        for (std::size_t i = 0; i < y.size(); ++i) {
            Rational value(y[i], block.vector[i]);
            if (less(result.value, value)) result.value = value;
        }
        result.blocks.push_back(std::move(block));
    }
    return result;
}
int popcount(U64 x) { return __builtin_popcountll(x); }
U64 cardinality(U64 a, U64 b) {
    int low = popcount(a), high = 64 - popcount(b);
    int s = std::min(high, std::max(low, 32));
    return static_cast<U64>(s * (64 - s));
}
int branch_edge(U64 a, U64 b) {
    Matrix l = matrix(FULL ^ b), r = matrix(FULL ^ a);
    std::array<int, N> li{}, lo{}, ri{}, ro{};
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        li[j] += l[i][j]; lo[i] += l[i][j]; ri[j] += r[i][j]; ro[i] += r[i][j];
    }
    int best = -1, bestscore = -1;
    for (int edge = 0; edge < 64; ++edge) if (!(((a | b) >> edge) & 1ULL)) {
        int i = edge / 4, j = edge % 16;
        int score = 2 * (li[i] * lo[j] + ri[i] * ro[j]) + (i == j);
        if (score > bestscore) { bestscore = score; best = edge; }
    }
    if (best < 0) throw std::logic_error("no unassigned branch edge");
    return best;
}
struct Node {
    U64 a = 0, b = 0;
    char kind = 'F'; // F frontier, S split, P pruned, C complete, Y colour symmetry
    bool collatz_proof = false;
    int edge = -1;
    std::size_t left = 0, right = 0;
};
std::string hex(U64 value) { std::ostringstream s; s << "0x" << std::hex << value; return s.str(); }
void output_block(std::ostream& out, const Bound& bound) {
    out << "{\"exponent\":" << EXPONENT << ",\"components\":[";
    bool first = true;
    for (const auto& block : bound.blocks) {
        if (!first) out << ',';
        first = false;
        out << "{\"vertices\":[";
        for (std::size_t j = 0; j < block.vertices.size(); ++j) { if (j) out << ','; out << block.vertices[j]; }
        out << "],\"vector\":[";
        for (std::size_t j = 0; j < block.vector.size(); ++j) { if (j) out << ','; out << block.vector[j]; }
        out << "]}";
    }
    out << "]}";
}
void atomic_state(const std::string& path, const std::vector<Node>& nodes, U64 processed) {
    std::ofstream out(path + ".tmp");
    if (!out) throw std::runtime_error("cannot write checkpoint state");
    out << "M3STATE 1 " << nodes.size() << ' ' << processed << '\n';
    for (const auto& node : nodes) out << node.a << ' ' << node.b << ' ' << node.kind << ' ' << node.collatz_proof << ' ' << node.edge << ' ' << node.left << ' ' << node.right << '\n';
    out.close();
    if (!out || std::rename((path + ".tmp").c_str(), path.c_str())) throw std::runtime_error("checkpoint atomic replacement failed");
}
void write_certificate(const std::string& path, const std::vector<Node>& nodes, const std::string& candidate,
                       U64 threshold_n, U64 threshold_d, U64 processed, double seconds, std::size_t budget) {
    std::ofstream out(path + ".tmp");
    if (!out) throw std::runtime_error("cannot write certificate");
    out << "{\"schema\":\"arbitrary-window-frontier-v1\",\"q\":4,\"m\":3,\"root\":0,"
           "\"word_order\":\"lexicographic base-4 integer; all 64 edge bits independent\",\"incumbent\":" << candidate
        << ",\"prune_threshold\":{\"numerator\":" << threshold_n << ",\"denominator\":" << threshold_d << "},\"nodes\":[";
    std::size_t frontier = 0, pruned = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (i) out << ',';
        const char* kind = node.kind == 'F' ? "frontier" : node.kind == 'S' ? "split" : node.kind == 'P' ? "pruned" : node.kind == 'C' ? "complete_assignment" : "symmetry";
        frontier += node.kind == 'F'; pruned += node.kind == 'P';
        out << "{\"id\":" << i << ",\"a\":\"" << hex(node.a) << "\",\"b\":\"" << hex(node.b) << "\",\"kind\":\"" << kind << "\"";
        if (node.kind == 'Y') {
            out << ",\"ref\":1,\"permutation\":[0,1,2,3],\"reversal\":false,\"colour_swap\":true}";
            continue;
        }
        if (node.kind == 'S') out << ",\"edge\":" << node.edge << ",\"children\":[" << node.left << ',' << node.right << ']';
        if (node.collatz_proof) {
            auto left = collatz(matrix(FULL ^ node.b)), right = collatz(matrix(FULL ^ node.a));
            Rational bound = multiply(left.value, right.value);
            out << ",\"proofs\":[{\"kind\":\"scc_collatz_product\",\"left\":";
            output_block(out, left); out << ",\"right\":"; output_block(out, right);
            out << "}],\"upper\":{\"numerator\":" << bound.numerator << ",\"denominator\":" << bound.denominator << ",\"power\":" << EXPONENT << "}}";
        } else {
            U64 bound = std::min<U64>(1023, cardinality(node.a, node.b));
            out << ",\"proofs\":[{\"kind\":\"cardinality\"},{\"kind\":\"dna_m3_integer_gap\"}],\"upper\":{\"numerator\":" << bound << ",\"denominator\":1,\"power\":3}}";
        }
    }
    out << "],\"stats\":{\"nodes_processed\":" << processed << ",\"stored_nodes\":" << nodes.size()
        << ",\"frontier_nodes\":" << frontier << ",\"pruned_nodes\":" << pruned
        << ",\"last_run_seconds\":" << std::setprecision(17) << seconds << "},\"run_config\":{\"producer\":\"native checked uint64/uint128\",\"collatz_exponent\":4,\"vector_steps\":8,\"cpu_limit\":1,\"max_nodes_this_run\":" << budget
        << ",\"memory_contract_bytes\":4294967296,\"memory_guard_bytes\":3221225472,\"compute\":\"existing local CPU; no cloud\"}}\n";
    out.close();
    if (!out || std::rename((path + ".tmp").c_str(), path.c_str())) throw std::runtime_error("certificate atomic replacement failed");
}

int main(int argc, char** argv) {
    try {
        std::string output, candidate_path, resume;
        std::size_t budget = 25000;
        double seconds_limit = 30;
        U64 threshold_n = 2253, threshold_d = 250;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 == argc) throw std::invalid_argument("every option requires a value");
            std::string value = argv[++i];
            if (arg == "--output") output = value;
            else if (arg == "--candidate") candidate_path = value;
            else if (arg == "--resume-state") resume = value;
            else if (arg == "--max-nodes") budget = std::stoull(value);
            else if (arg == "--seconds") seconds_limit = std::stod(value);
            else if (arg == "--threshold-numerator") threshold_n = std::stoull(value);
            else if (arg == "--threshold-denominator") threshold_d = std::stoull(value);
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (output.empty() || candidate_path.empty() || budget > 500000 || !std::isfinite(seconds_limit) || seconds_limit < 0 || seconds_limit > 120 || threshold_n > 100000 || !threshold_d || threshold_d > 100000)
            throw std::invalid_argument("pilot requires output,candidate,nodes<=500000,seconds<=120 and bounded positive rational threshold");
        std::ifstream cand_file(candidate_path);
        if (!cand_file) throw std::runtime_error("candidate certificate missing");
        std::string candidate((std::istreambuf_iterator<char>(cand_file)), std::istreambuf_iterator<char>());
        std::vector<Node> nodes;
        U64 processed = 0;
        if (resume.empty()) {
            nodes = {Node{0, 0, 'S', false, 0, 1, 2}, Node{1, 0}, Node{0, 1, 'Y'}};
        } else {
            std::ifstream state(resume);
            std::string magic; int version; std::size_t count;
            state >> magic >> version >> count >> processed;
            if (!state || magic != "M3STATE" || version != 1 || count > 2000000) throw std::runtime_error("invalid native checkpoint header");
            nodes.resize(count);
            for (auto& node : nodes) state >> node.a >> node.b >> node.kind >> node.collatz_proof >> node.edge >> node.left >> node.right;
            if (!state) throw std::runtime_error("truncated native checkpoint");
        }
        nodes.reserve(nodes.size() + 2 * budget + 2);
        std::vector<std::size_t> queue;
        for (std::size_t i = nodes.size(); i-- > 0;) if (nodes[i].kind == 'F') queue.push_back(i);
        std::signal(SIGINT, signal_stop); std::signal(SIGTERM, signal_stop);
        auto start = std::chrono::steady_clock::now();
        std::size_t count = 0;
        auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
        while (!queue.empty() && count < budget && elapsed() < seconds_limit && !stopped) {
            if (count % 1024 == 0) {
                struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
                U64 rss = static_cast<U64>(usage.ru_maxrss);
#ifndef __APPLE__
                rss *= 1024;
#endif
                if (rss > 3ULL * 1024 * 1024 * 1024) break;
            }
            std::size_t i = queue.back(); queue.pop_back();
            Node node = nodes[i];
            if ((node.a & node.b) || node.kind != 'F') throw std::logic_error("corrupt native frontier");
            ++count; ++processed;
            if (below_target(Rational(cardinality(node.a, node.b), 1), 3, threshold_n, threshold_d)) {
                nodes[i].kind = 'P'; continue;
            }
            Bound left = collatz(matrix(FULL ^ node.b)), right = collatz(matrix(FULL ^ node.a));
            if (below_target(multiply(left.value, right.value), EXPONENT, threshold_n, threshold_d)) {
                nodes[i].kind = 'P'; nodes[i].collatz_proof = true; continue;
            }
            if ((node.a | node.b) == FULL) { nodes[i].kind = 'C'; nodes[i].collatz_proof = true; continue; }
            int edge = branch_edge(node.a, node.b);
            U64 bit = 1ULL << edge;
            std::size_t first = nodes.size();
            // A frontier disappears only after both complementary children exist.
            nodes.push_back(Node{node.a | bit, node.b}); nodes.push_back(Node{node.a, node.b | bit});
            nodes[i].kind = 'S'; nodes[i].edge = edge; nodes[i].left = first; nodes[i].right = first + 1;
            queue.push_back(first + 1); queue.push_back(first);
        }
        double duration = elapsed();
        atomic_state(output + ".state", nodes, processed);
        write_certificate(output, nodes, candidate, threshold_n, threshold_d, processed, duration, budget);
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        std::cout << "{\"status\":\"PRODUCED_UNVERIFIED\",\"nodes_this_run\":" << count
                  << ",\"nodes_processed\":" << processed << ",\"stored_nodes\":" << nodes.size()
                  << ",\"wall_seconds\":" << std::setprecision(10) << duration
                  << ",\"nodes_per_second\":" << (duration ? count / duration : 0)
                  << ",\"maxrss_platform_units\":" << usage.ru_maxrss
                  << ",\"interrupted\":" << (stopped ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "certificate producer refused: " << error.what() << '\n';
        return 1;
    }
}
