#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int Q = 4;
constexpr int MAX_WORDS = 64;  // supports m <= 3, enough for the present m=2 audit

struct Args {
    int m = 2;
    std::vector<int> n_values{1000, 5000, 10000};
    std::vector<std::string> regimes{"fixed"};
    int fixed_k = 20;
    int particles = 2000;
    int reps = 3;
    int jobs = 4;
    int seed = 2026070702;
    double capacity_reference = 1.167987028946;
    std::string out_dir;
};

struct State {
    int suffix = 0;
    int suffix_len = 0;
    std::array<int, MAX_WORDS> first{};
};

struct StateHash {
    int word_count = 16;
    std::size_t operator()(State const& s) const noexcept {
        std::size_t h = static_cast<std::size_t>(s.suffix * 1315423911u + s.suffix_len);
        for (int i = 0; i < word_count; ++i) {
            h ^= static_cast<std::size_t>(s.first[i] + 0x9e3779b9 + (h << 6) + (h >> 2));
        }
        return h;
    }
};

struct StateEq {
    int word_count = 16;
    bool operator()(State const& a, State const& b) const noexcept {
        if (a.suffix != b.suffix || a.suffix_len != b.suffix_len) return false;
        for (int i = 0; i < word_count; ++i) {
            if (a.first[i] != b.first[i]) return false;
        }
        return true;
    }
};

struct RawRow {
    int m = 2;
    int k = 0;
    int n = 0;
    int rep = 0;
    std::uint64_t seed = 0;
    int particles = 0;
    double log2_count_est = 0.0;
    double rate_est_bits_per_nt = 0.0;
    double mean_branching_est = 0.0;
    int unique_particle_states = 0;
    std::string run_status = "ok";
    int extinction_n = -1;
    double elapsed_sec = 0.0;
    std::string regime;
};

struct Task {
    int m;
    int k;
    int particles;
    int rep;
    int n;
    std::uint64_t seed;
    std::string regime;
};

std::vector<std::string> split_list(std::string text) {
    for (char& c : text) {
        if (c == ',') c = ' ';
    }
    std::stringstream ss(text);
    std::vector<std::string> out;
    std::string item;
    while (ss >> item) out.push_back(item);
    return out;
}

std::vector<int> parse_int_list(std::string const& text) {
    std::vector<int> out;
    for (auto const& item : split_list(text)) out.push_back(std::stoi(item));
    return out;
}

int reverse_complement(int word, int m) {
    int rc = 0;
    for (int i = 0; i < m; ++i) {
        int base = word & 3;
        word >>= 2;
        rc = (rc << 2) | (3 - base);
    }
    return rc;
}

int k_for_regime(std::string const& regime, int n, int fixed_k) {
    auto robust_floor = [](double value) {
        return static_cast<int>(std::floor(std::nextafter(value, std::numeric_limits<double>::infinity())));
    };
    if (regime == "fixed") return fixed_k;
    if (regime == "sqrt") return robust_floor(std::sqrt(static_cast<double>(n)));
    if (regime == "n067" || regime == "n2over3") return robust_floor(std::pow(static_cast<double>(n), 2.0 / 3.0));
    if (regime == "n075") return robust_floor(std::pow(static_cast<double>(n), 0.75));
    if (regime == "n09" || regime == "n0.9") return robust_floor(std::pow(static_cast<double>(n), 0.90));
    if (regime == "nlog" || regime == "n_over_log" || regime == "n/log") {
        return robust_floor(static_cast<double>(n) / std::log(static_cast<double>(std::max(n, 3))));
    }
    if (regime.rfind("linear", 0) == 0) {
        double alpha = std::stod(regime.substr(6));
        return robust_floor(alpha * n);
    }
    throw std::runtime_error("unknown regime: " + regime);
}

std::string fmt(double x, int prec = 12) {
    if (!std::isfinite(x)) return "-inf";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << x;
    return ss.str();
}

int valid_mask_for_state(
    State const& st,
    int m,
    int word_count,
    int word_mask,
    int min_start_separation,
    int start_pos,
    std::vector<int> const& rc_table
) {
    (void)word_count;
    if (st.suffix_len < m - 1) return 0b1111;
    int old_cutoff = start_pos - min_start_separation;
    int mask = 0;
    for (int base = 0; base < Q; ++base) {
        int word = ((st.suffix << 2) | base) & word_mask;
        int rc = rc_table[word];
        if (st.first[rc] <= old_cutoff) continue;
        mask |= (1 << base);
    }
    return mask;
}

int popcount4(int mask) {
    return ((mask >> 0) & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1);
}

int choose_base_from_mask(int mask, int choice_index) {
    for (int base = 0; base < Q; ++base) {
        if (mask & (1 << base)) {
            if (choice_index == 0) return base;
            --choice_index;
        }
    }
    return 0;
}

State extend_state(
    State const& st,
    int base,
    int m,
    int word_mask,
    int suffix_mask,
    int start_pos,
    int inf_pos
) {
    State child = st;
    if (st.suffix_len < m - 1) {
        child.suffix = (st.suffix << 2) | base;
        child.suffix_len = st.suffix_len + 1;
        return child;
    }
    int word = ((st.suffix << 2) | base) & word_mask;
    if (child.first[word] == inf_pos) child.first[word] = start_pos;
    child.suffix = (m > 1) ? (word & suffix_mask) : 0;
    child.suffix_len = std::min(m - 1, st.suffix_len + 1);
    return child;
}

RawRow run_task(Task const& task) {
    if (task.m < 1 || task.m > 3) throw std::runtime_error("C++ fast SMC supports 1 <= m <= 3");
    int word_count = 1 << (2 * task.m);
    int word_mask = (1 << (2 * task.m)) - 1;
    int suffix_mask = (task.m > 1) ? ((1 << (2 * (task.m - 1))) - 1) : 0;
    int min_start_separation = task.m + task.k;
    int inf_pos = task.n + min_start_separation + 1;
    std::vector<int> rc_table(word_count);
    for (int w = 0; w < word_count; ++w) rc_table[w] = reverse_complement(w, task.m);

    State init;
    init.suffix = 0;
    init.suffix_len = 0;
    init.first.fill(inf_pos);
    std::vector<State> states(task.particles, init), next_states(task.particles);
    std::vector<int> masks(task.particles, 0), weights(task.particles, 0);

    std::mt19937_64 rng(task.seed);
    double log2_count = 0.0;
    double mean_branching = 0.0;
    int extinction_n = -1;
    auto t0 = std::chrono::steady_clock::now();

    for (int len = 1; len <= task.n; ++len) {
        int start_pos = len - task.m;
        int total_weight = 0;
        for (int i = 0; i < task.particles; ++i) {
            int mask = valid_mask_for_state(states[i], task.m, word_count, word_mask, min_start_separation, start_pos, rc_table);
            int weight = popcount4(mask);
            masks[i] = mask;
            weights[i] = weight;
            total_weight += weight;
        }

        if (total_weight <= 0) {
            log2_count = -INFINITY;
            mean_branching = 0.0;
            extinction_n = len;
            states.clear();
            break;
        }

        mean_branching = static_cast<double>(total_weight) / static_cast<double>(task.particles);
        log2_count += std::log2(mean_branching);

        double spacing = static_cast<double>(total_weight) / static_cast<double>(task.particles);
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        double threshold = uniform01(rng) * spacing;
        int parent = 0;
        double cumulative = static_cast<double>(weights[0]);
        for (int i = 0; i < task.particles; ++i) {
            while (threshold >= cumulative && parent + 1 < task.particles) {
                ++parent;
                cumulative += static_cast<double>(weights[parent]);
            }
            int w = weights[parent];
            std::uniform_int_distribution<int> choice_dist(0, w - 1);
            int base = choose_base_from_mask(masks[parent], choice_dist(rng));
            next_states[i] = extend_state(states[parent], base, task.m, word_mask, suffix_mask, start_pos, inf_pos);
            threshold += spacing;
        }
        states.swap(next_states);
    }

    StateHash hasher;
    hasher.word_count = word_count;
    StateEq eq;
    eq.word_count = word_count;
    std::unordered_set<State, StateHash, StateEq> uniq(0, hasher, eq);
    uniq.reserve(static_cast<std::size_t>(task.particles) * 2);
    for (auto const& st : states) uniq.insert(st);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    RawRow row;
    row.m = task.m;
    row.k = task.k;
    row.n = task.n;
    row.rep = task.rep;
    row.seed = task.seed;
    row.particles = task.particles;
    row.log2_count_est = log2_count;
    row.rate_est_bits_per_nt = log2_count / static_cast<double>(task.n);
    row.mean_branching_est = mean_branching;
    row.unique_particle_states = static_cast<int>(uniq.size());
    row.run_status = extinction_n < 0 ? "ok" : "particle_extinction";
    row.extinction_n = extinction_n;
    row.elapsed_sec = elapsed;
    row.regime = task.regime;
    return row;
}

void ensure_dir(std::string const& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) throw std::runtime_error("failed to create output directory: " + path + ": " + error.message());
}

void write_raw(std::string const& path, std::vector<RawRow> const& rows) {
    std::ofstream out(path);
    out << "m,k,n,rep,seed,particles,log2_count_est,rate_est_bits_per_nt,mean_branching_est,unique_particle_states,state_representation,transition_cache_size,transition_cache_limit,run_status,extinction_n,elapsed_sec,regime\n";
    for (auto const& r : rows) {
        out << r.m << ',' << r.k << ',' << r.n << ',' << r.rep << ',' << r.seed << ',' << r.particles << ','
            << fmt(r.log2_count_est) << ',' << fmt(r.rate_est_bits_per_nt) << ',' << fmt(r.mean_branching_est) << ','
            << r.unique_particle_states << ",first_seen_cpp,0,not_applicable," << r.run_status << ',';
        if (r.extinction_n >= 0) out << r.extinction_n;
        out << ',' << std::fixed << std::setprecision(3) << r.elapsed_sec << ',' << r.regime << '\n';
    }
}

void write_summary(std::string const& path, std::vector<RawRow> const& rows, double capacity_reference) {
    using Key = std::tuple<int, std::string, int, int, int>;  // m, regime, k, n, particles
    std::map<Key, std::vector<RawRow const*>> groups;
    for (auto const& r : rows) {
        groups[{r.m, r.regime, r.k, r.n, r.particles}].push_back(&r);
    }
    std::ofstream out(path);
    out << "m,regime,k,n,reps,finite_reps,failed_reps,particles,mean_rate_est_bits_per_nt,sd_rate,sem_rate,excess_over_capacity,capacity_reference,summary_status\n";
    for (auto const& item : groups) {
        auto const& key = item.first;
        auto const& group_rows = item.second;
        std::vector<double> finite_vals;
        for (auto const* row : group_rows) {
            if (row->run_status == "ok" && std::isfinite(row->rate_est_bits_per_nt)) {
                finite_vals.push_back(row->rate_est_bits_per_nt);
            }
        }
        std::size_t failed_reps = group_rows.size() - finite_vals.size();
        int m, k, n, particles;
        std::string regime;
        std::tie(m, regime, k, n, particles) = key;
        out << m << ',' << regime << ',' << k << ',' << n << ',' << group_rows.size() << ','
            << finite_vals.size() << ',' << failed_reps << ',' << particles << ',';
        if (failed_reps == 0) {
            double mean = std::accumulate(finite_vals.begin(), finite_vals.end(), 0.0) /
                static_cast<double>(finite_vals.size());
            double sd = 0.0;
            if (finite_vals.size() > 1) {
                double ss = 0.0;
                for (double v : finite_vals) ss += (v - mean) * (v - mean);
                sd = std::sqrt(ss / static_cast<double>(finite_vals.size() - 1));
            }
            double sem = finite_vals.size() > 1 ? sd / std::sqrt(static_cast<double>(finite_vals.size())) : 0.0;
            out << fmt(mean) << ',' << fmt(sd) << ',' << fmt(sem) << ','
                << fmt(mean - capacity_reference) << ',' << fmt(capacity_reference) << ",ok\n";
        } else {
            // Do not average finite survivors after any replicate has become
            // extinct: that would turn an estimator failure into a plausible
            // scientific rate estimate.
            out << ",,,," << fmt(capacity_reference) << ",particle_extinction\n";
        }
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](std::string const& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (a == "--m") args.m = std::stoi(need(a));
        else if (a == "--n-values") args.n_values = parse_int_list(need(a));
        else if (a == "--regimes") args.regimes = split_list(need(a));
        else if (a == "--fixed-k") args.fixed_k = std::stoi(need(a));
        else if (a == "--particles") args.particles = std::stoi(need(a));
        else if (a == "--reps") args.reps = std::stoi(need(a));
        else if (a == "--jobs") args.jobs = std::stoi(need(a));
        else if (a == "--seed") args.seed = std::stoi(need(a));
        else if (a == "--capacity-reference") args.capacity_reference = std::stod(need(a));
        else if (a == "--out-dir") args.out_dir = need(a);
        else if (a == "--cache-max") (void)need(a);  // accepted for CLI compatibility
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (args.out_dir.empty()) throw std::runtime_error("--out-dir is required");
    args.jobs = std::max(1, args.jobs);
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        bool has_growing_threshold = std::any_of(
            args.regimes.begin(), args.regimes.end(),
            [](std::string const& regime) { return regime != "fixed"; }
        );
        if (has_growing_threshold) {
            std::cerr
                << "WARNING: growing-k SMC is experimental and can suffer complete "
                << "particle degeneracy. Do not interpret non-finite or collapse-dominated "
                << "rates as capacity estimates. Fixed-k runs are the stable default."
                << std::endl;
        }
        std::vector<Task> tasks;
        for (std::size_t regime_idx = 0; regime_idx < args.regimes.size(); ++regime_idx) {
            auto const& regime = args.regimes[regime_idx];
            for (int n : args.n_values) {
                int k = k_for_regime(regime, n, args.fixed_k);
                for (int rep = 0; rep < args.reps; ++rep) {
                    std::uint64_t seed = static_cast<std::uint64_t>(args.seed) + 100000ull * regime_idx + 997ull * static_cast<std::uint64_t>(n) + 17ull * static_cast<std::uint64_t>(rep);
                    tasks.push_back(Task{args.m, k, args.particles, rep, n, seed, regime});
                }
            }
        }

        std::cout << "Running C++ first-seen regime SMC: m=" << args.m << ", tasks=" << tasks.size()
                  << ", particles=" << args.particles << ", reps=" << args.reps
                  << ", jobs=" << args.jobs << std::endl;

        std::vector<RawRow> rows(tasks.size());
        std::atomic<std::size_t> next{0};
        std::mutex io_mu;
        int workers = std::min<int>(args.jobs, static_cast<int>(tasks.size()));
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (int t = 0; t < workers; ++t) {
            threads.emplace_back([&]() {
                while (true) {
                    std::size_t idx = next.fetch_add(1);
                    if (idx >= tasks.size()) break;
                    rows[idx] = run_task(tasks[idx]);
                    {
                        std::lock_guard<std::mutex> lock(io_mu);
                        std::cout << "TASK_DONE " << (idx + 1) << "/" << tasks.size()
                                  << " regime=" << tasks[idx].regime
                                  << " n=" << tasks[idx].n
                                  << " k=" << tasks[idx].k
                                  << " rep=" << tasks[idx].rep
                                  << " rate=" << fmt(rows[idx].rate_est_bits_per_nt)
                                  << " elapsed=" << std::fixed << std::setprecision(3) << rows[idx].elapsed_sec
                                  << std::endl;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();

        auto degenerate = std::count_if(
            rows.begin(), rows.end(),
            [](RawRow const& row) { return !std::isfinite(row.rate_est_bits_per_nt); }
        );
        if (degenerate > 0) {
            std::cerr
                << "WARNING: particle degeneracy produced " << degenerate
                << " non-finite replicate(s). Raw failure rows and a non-numeric "
                << "failure summary will be written; no failed group is averaged."
                << std::endl;
        }

        std::sort(rows.begin(), rows.end(), [](RawRow const& a, RawRow const& b) {
            return std::tie(a.regime, a.n, a.rep) < std::tie(b.regime, b.n, b.rep);
        });

        ensure_dir(args.out_dir);
        std::string raw_path = args.out_dir + "/smc_regime_m" + std::to_string(args.m) + "_raw.csv";
        std::string summary_path = args.out_dir + "/smc_regime_m" + std::to_string(args.m) + "_summary.csv";
        write_raw(raw_path, rows);
        write_summary(summary_path, rows, args.capacity_reference);
        std::cout << raw_path << std::endl;
        std::cout << summary_path << std::endl;
        if (degenerate > 0) {
            std::cerr << "REGIME_SMC_PARTICLE_EXTINCTION" << std::endl;
            return 2;
        }
        std::cout << "REGIME_SMC_DONE" << std::endl;
    } catch (std::exception const& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
