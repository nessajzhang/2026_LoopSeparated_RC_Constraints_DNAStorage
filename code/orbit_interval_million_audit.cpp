// Deterministic high-volume stress audit for the orbit-interval lemma.
//
// This program works at the occurrence-position level.  That abstraction is
// deliberately broader than realizable DNA words: the lemma uses only the
// positions occupied by one reverse-complement orbit, so arbitrary position
// sets can expose a geometric counterexample even when no word realizes them.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kDefaultCases = 6'000'000;
constexpr std::uint64_t kDefaultSeed = 20'260'712;

constexpr std::array<std::int64_t, 27> kMValues = {
    1,   2,   3,   4,   5,   6,   8,    10,   12,
    16,  24,  32,  48,  64,  96,  128,  192,  256,
    384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192};

// Number of length-m window starts.  Zero covers n < m; the largest value
// probes geometry corresponding to sequences beyond the million-symbol scale.
constexpr std::array<std::int64_t, 18> kStartCounts = {
    0,      1,       2,       4,       8,       16,
    32,     64,      128,     256,     512,     1'000,
    10'000, 100'000, 500'000, 1'000'000, 5'000'000, 10'000'000};

constexpr std::array<const char*, 16> kScheduleNames = {
    "k_zero",       "k_fixed1",      "k_fixed4",      "k_fixed20",
    "k_log2_n",     "k_sqrt_n",      "k_n_0.60",      "k_n_0.75",
    "k_n_over_log", "k_linear_0.01", "k_linear_0.10", "k_linear_0.25",
    "k_linear_0.50", "k_linear_0.90", "k_linear_0.99", "k_maximal"};

constexpr std::array<const char*, 12> kModeNames = {
    "pair_random",          "pair_maximal_valid", "pair_boundary_invalid",
    "self_random",          "self_maximal_valid", "self_boundary_invalid",
    "one_side_empty",       "vacuous_full_span",  "dense_valid",
    "left_boundary_anchor", "right_boundary_anchor", "pair_two_anchor_valid"};

struct SplitMix64 {
  std::uint64_t state;

  std::uint64_t next() {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
  }

  std::int64_t uniform(std::int64_t upper_exclusive) {
    if (upper_exclusive <= 1) return 0;
    return static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(upper_exclusive));
  }
};

struct Aggregate {
  std::uint64_t cases = 0;
  std::uint64_t passed = 0;
  std::uint64_t valid = 0;
  std::uint64_t invalid = 0;
  std::uint64_t assertions = 0;
  std::uint64_t tight_pair = 0;
  std::uint64_t tight_self = 0;
  long double max_pair_ratio = 0.0L;
  long double max_self_ratio = 0.0L;
};

struct ThreadStats {
  std::vector<Aggregate> strata;
  std::uint64_t failures = 0;
  std::string first_failure;
};

struct Config {
  std::uint64_t cases = kDefaultCases;
  std::uint64_t seed = kDefaultSeed;
  unsigned threads = 1;
  std::string summary_path = "output/orbit_interval_million_audit_summary.json";
  std::string csv_path = "output/orbit_interval_million_audit_strata.csv";
};

std::int64_t clamp_k(std::int64_t k, std::int64_t n) {
  return std::clamp<std::int64_t>(k, 0, std::max<std::int64_t>(0, n - 1));
}

std::int64_t schedule_k(std::size_t schedule, std::int64_t n) {
  const long double x = static_cast<long double>(std::max<std::int64_t>(1, n));
  std::int64_t value = 0;
  switch (schedule) {
    case 0: value = 0; break;
    case 1: value = 1; break;
    case 2: value = 4; break;
    case 3: value = 20; break;
    case 4: value = static_cast<std::int64_t>(std::floor(std::log2(x))); break;
    case 5: value = static_cast<std::int64_t>(std::floor(std::sqrt(x))); break;
    case 6: value = static_cast<std::int64_t>(std::floor(std::pow(x, 0.60L))); break;
    case 7: value = static_cast<std::int64_t>(std::floor(std::pow(x, 0.75L))); break;
    case 8:
      value = static_cast<std::int64_t>(std::floor(x / std::max<long double>(1.0L, std::log(x))));
      break;
    case 9: value = static_cast<std::int64_t>(std::floor(0.01L * x)); break;
    case 10: value = static_cast<std::int64_t>(std::floor(0.10L * x)); break;
    case 11: value = static_cast<std::int64_t>(std::floor(0.25L * x)); break;
    case 12: value = static_cast<std::int64_t>(std::floor(0.50L * x)); break;
    case 13: value = static_cast<std::int64_t>(std::floor(0.90L * x)); break;
    case 14: value = static_cast<std::int64_t>(std::floor(0.99L * x)); break;
    default: value = std::max<std::int64_t>(0, n - 1); break;
  }
  return clamp_k(value, n);
}

void normalize(std::vector<std::int64_t>& positions) {
  std::sort(positions.begin(), positions.end());
  positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
}

std::int64_t random_inclusive(SplitMix64& rng, std::int64_t lo, std::int64_t hi) {
  if (hi <= lo) return lo;
  return lo + rng.uniform(hi - lo + 1);
}

bool pair_valid_direct(const std::vector<std::int64_t>& a,
                       const std::vector<std::int64_t>& b,
                       std::int64_t distance,
                       std::uint64_t& assertions) {
  for (const auto x : a) {
    for (const auto y : b) {
      ++assertions;
      if (std::llabs(x - y) >= distance) return false;
    }
  }
  return true;
}

bool pair_valid_extrema(const std::vector<std::int64_t>& a,
                        const std::vector<std::int64_t>& b,
                        std::int64_t distance) {
  if (a.empty() || b.empty()) return true;
  const auto [amin, amax] = std::minmax_element(a.begin(), a.end());
  const auto [bmin, bmax] = std::minmax_element(b.begin(), b.end());
  return std::max(*amax - *bmin, *bmax - *amin) < distance;
}

bool self_valid_direct(const std::vector<std::int64_t>& positions,
                       std::int64_t distance,
                       std::uint64_t& assertions) {
  for (std::size_t i = 0; i < positions.size(); ++i) {
    for (std::size_t j = i + 1; j < positions.size(); ++j) {
      ++assertions;
      if (positions[j] - positions[i] >= distance) return false;
    }
  }
  return true;
}

bool self_valid_extrema(const std::vector<std::int64_t>& positions,
                        std::int64_t distance) {
  return positions.size() < 2 || positions.back() - positions.front() < distance;
}

void random_positions(std::vector<std::int64_t>& out, SplitMix64& rng,
                      std::int64_t starts, int maximum_count = 24) {
  if (starts <= 0) return;
  const int count = 1 + static_cast<int>(rng.uniform(maximum_count));
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) out.push_back(rng.uniform(starts));
  normalize(out);
}

void interval_positions(std::vector<std::int64_t>& out, SplitMix64& rng,
                        std::int64_t lo, std::int64_t hi, int maximum_count = 24) {
  if (hi < lo) return;
  out.push_back(lo);
  out.push_back(hi);
  const int extra = static_cast<int>(rng.uniform(maximum_count - 1));
  for (int i = 0; i < extra; ++i) out.push_back(random_inclusive(rng, lo, hi));
  normalize(out);
}

void generate_case(std::size_t mode, std::int64_t starts, std::int64_t distance,
                   SplitMix64& rng, std::vector<std::int64_t>& a,
                   std::vector<std::int64_t>& b, bool& singleton) {
  singleton = mode >= 3 && mode <= 5;
  if (starts <= 0) return;

  const std::int64_t last = starts - 1;
  if (mode == 0 || mode == 3) {
    random_positions(a, rng, starts);
    if (!singleton) random_positions(b, rng, starts);
    return;
  }

  if (mode == 1) {
    const std::int64_t p = rng.uniform(starts);
    a = {p};
    interval_positions(b, rng, std::max<std::int64_t>(0, p - distance + 1),
                       std::min(last, p + distance - 1));
    return;
  }
  if (mode == 2) {
    if (distance < starts) {
      if (rng.next() & 1ULL) {
        const std::int64_t p = rng.uniform(starts - distance);
        a = {p};
        b = {p + distance};
      } else {
        const std::int64_t p = distance + rng.uniform(starts - distance);
        a = {p};
        b = {p - distance};
      }
    } else {
      a = {0};
      b = {last};
    }
    return;
  }
  if (mode == 4) {
    const std::int64_t lo = rng.uniform(starts);
    const std::int64_t hi = std::min(last, lo + distance - 1);
    interval_positions(a, rng, lo, hi);
    return;
  }
  if (mode == 5) {
    if (distance < starts) {
      a = {0, distance};
    } else {
      a = {0, last};
    }
    return;
  }
  if (mode == 6) {
    random_positions(a, rng, starts);
    return;
  }
  if (mode == 7) {
    a = {0, last};
    b = {0, last};
    return;
  }
  if (mode == 8) {
    const std::int64_t width = std::min(last, distance - 1);
    const std::int64_t base = rng.uniform(starts - width);
    interval_positions(a, rng, base, base + width);
    interval_positions(b, rng, base, base + width);
    return;
  }
  if (mode == 9 || mode == 10) {
    const std::int64_t p = mode == 9 ? 0 : last;
    a = {p};
    interval_positions(b, rng, std::max<std::int64_t>(0, p - distance + 1),
                       std::min(last, p + distance - 1));
    return;
  }

  // Two selected-side anchors and a non-core cluster in their common
  // distance-(D-1) neighborhood.  This exercises the fact that any selected
  // occurrence may serve as the proof anchor.
  const std::int64_t width = std::min(last, distance - 1);
  const std::int64_t base = rng.uniform(starts - width);
  a = {base, base + width};
  interval_positions(b, rng, base, base + width);
}

std::size_t stratum_index(std::size_t mode, std::size_t schedule,
                          std::size_t m_index, std::size_t n_index) {
  return (((mode * kScheduleNames.size() + schedule) * kMValues.size() + m_index) *
          kStartCounts.size() + n_index);
}

bool audit_case(std::uint64_t case_id, const Config& config, Aggregate& aggregate,
                std::string& failure) {
  std::uint64_t selector = case_id;
  const std::size_t mode = selector % kModeNames.size();
  selector /= kModeNames.size();
  const std::size_t schedule = selector % kScheduleNames.size();
  selector /= kScheduleNames.size();
  const std::size_t m_index = selector % kMValues.size();
  selector /= kMValues.size();
  const std::size_t n_index = selector % kStartCounts.size();

  const std::int64_t m = kMValues[m_index];
  const std::int64_t starts = kStartCounts[n_index];
  const std::int64_t n = starts == 0 ? std::max<std::int64_t>(0, m - 1) : starts + m - 1;
  const std::int64_t k = schedule_k(schedule, n);
  const std::int64_t distance = m + k;
  SplitMix64 rng{config.seed ^ (case_id * 0xd6e8feb86659fd93ULL)};

  std::vector<std::int64_t> a;
  std::vector<std::int64_t> b;
  bool singleton = false;
  generate_case(mode, starts, distance, rng, a, b, singleton);
  normalize(a);
  normalize(b);

  ++aggregate.cases;
  std::uint64_t assertions = 0;
  bool valid = false;
  if (singleton) {
    const bool direct = self_valid_direct(a, distance, assertions);
    const bool extrema = self_valid_extrema(a, distance);
    ++assertions;
    if (direct != extrema) {
      failure = "self direct/extrema disagreement";
      return false;
    }
    valid = direct;
    if (valid && !a.empty()) {
      const std::int64_t span = a.back() - a.front() + 1;
      ++assertions;
      if (span > distance) {
        failure = "self-RC cover exceeds D";
        return false;
      }
      aggregate.max_self_ratio =
          std::max(aggregate.max_self_ratio,
                   static_cast<long double>(span) / static_cast<long double>(distance));
      if (span == distance) ++aggregate.tight_self;
      const std::int64_t expanded = span + m - 1;
      ++assertions;
      if (expanded > distance + m - 1) {
        failure = "self-RC marked interval exceeds D+m-1";
        return false;
      }
    }
  } else {
    const bool direct = pair_valid_direct(a, b, distance, assertions);
    const bool extrema = pair_valid_extrema(a, b, distance);
    ++assertions;
    if (direct != extrema) {
      failure = "pair direct/extrema disagreement";
      return false;
    }
    valid = direct;
    if (valid && !a.empty() && !b.empty()) {
      const std::int64_t span = b.back() - b.front() + 1;
      ++assertions;
      if (span > 2 * distance - 1) {
        failure = "paired non-core cover exceeds 2D-1";
        return false;
      }
      aggregate.max_pair_ratio =
          std::max(aggregate.max_pair_ratio,
                   static_cast<long double>(span) /
                       static_cast<long double>(2 * distance - 1));
      if (span == 2 * distance - 1) ++aggregate.tight_pair;
      for (const auto anchor : a) {
        ++assertions;
        if (b.front() < anchor - distance + 1 || b.back() > anchor + distance - 1) {
          failure = "valid pair is not covered from every selected anchor";
          return false;
        }
      }
      const std::int64_t expanded = span + m - 1;
      ++assertions;
      if (expanded > 2 * distance + m - 2) {
        failure = "paired marked interval exceeds 2D+m-2";
        return false;
      }
    }
  }

  if (valid) {
    ++aggregate.valid;
  } else {
    ++aggregate.invalid;
    // Every rejected case must expose an explicit pair at separation >= D.
    bool witness = false;
    if (singleton) {
      for (std::size_t i = 0; i < a.size() && !witness; ++i) {
        for (std::size_t j = i + 1; j < a.size(); ++j) {
          if (a[j] - a[i] >= distance) {
            witness = true;
            break;
          }
        }
      }
    } else {
      for (const auto x : a) {
        for (const auto y : b) {
          if (std::llabs(x - y) >= distance) {
            witness = true;
            break;
          }
        }
        if (witness) break;
      }
    }
    ++assertions;
    if (!witness) {
      failure = "invalid case lacks a separation witness";
      return false;
    }
  }

  // The two adversarial boundary modes must be invalid whenever a distance-D
  // pair fits in the start domain; otherwise the constraint is vacuous there.
  if ((mode == 2 || mode == 5) && distance < starts) {
    ++assertions;
    if (valid) {
      failure = "boundary-D construction was incorrectly accepted";
      return false;
    }
  }
  if (mode == 7 && distance > starts - 1) {
    ++assertions;
    if (!valid) {
      failure = "vacuous full-span construction was incorrectly rejected";
      return false;
    }
  }

  aggregate.assertions += assertions;
  ++aggregate.passed;
  return true;
}

Config parse_args(int argc, char** argv) {
  Config config;
  const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
  config.threads = std::max(1U, hardware > 2 ? hardware - 2 : hardware);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (++i >= argc) {
        std::cerr << "missing value for " << name << '\n';
        std::exit(2);
      }
      return argv[i];
    };
    if (arg == "--cases") config.cases = std::stoull(require_value("--cases"));
    else if (arg == "--seed") config.seed = std::stoull(require_value("--seed"));
    else if (arg == "--threads") config.threads = std::stoul(require_value("--threads"));
    else if (arg == "--summary") config.summary_path = require_value("--summary");
    else if (arg == "--csv") config.csv_path = require_value("--csv");
    else {
      std::cerr << "unknown argument: " << arg << '\n';
      std::exit(2);
    }
  }
  config.threads = std::max(1U, config.threads);
  return config;
}

void merge(Aggregate& target, const Aggregate& source) {
  target.cases += source.cases;
  target.passed += source.passed;
  target.valid += source.valid;
  target.invalid += source.invalid;
  target.assertions += source.assertions;
  target.tight_pair += source.tight_pair;
  target.tight_self += source.tight_self;
  target.max_pair_ratio = std::max(target.max_pair_ratio, source.max_pair_ratio);
  target.max_self_ratio = std::max(target.max_self_ratio, source.max_self_ratio);
}

}  // namespace

int main(int argc, char** argv) {
  const Config config = parse_args(argc, argv);
  const std::size_t strata_count = kModeNames.size() * kScheduleNames.size() *
                                   kMValues.size() * kStartCounts.size();
  std::vector<ThreadStats> thread_stats(config.threads);
  for (auto& stats : thread_stats) stats.strata.resize(strata_count);

  const auto start_time = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(config.threads);
  for (unsigned thread_id = 0; thread_id < config.threads; ++thread_id) {
    workers.emplace_back([&, thread_id]() {
      auto& stats = thread_stats[thread_id];
      for (std::uint64_t case_id = thread_id; case_id < config.cases;
           case_id += config.threads) {
        std::uint64_t selector = case_id;
        const std::size_t mode = selector % kModeNames.size();
        selector /= kModeNames.size();
        const std::size_t schedule = selector % kScheduleNames.size();
        selector /= kScheduleNames.size();
        const std::size_t m_index = selector % kMValues.size();
        selector /= kMValues.size();
        const std::size_t n_index = selector % kStartCounts.size();
        auto& aggregate = stats.strata[stratum_index(mode, schedule, m_index, n_index)];
        std::string failure;
        if (!audit_case(case_id, config, aggregate, failure)) {
          ++stats.failures;
          if (stats.first_failure.empty()) {
            stats.first_failure = "case=" + std::to_string(case_id) + " " + failure;
          }
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();

  std::vector<Aggregate> merged(strata_count);
  std::uint64_t failures = 0;
  std::string first_failure;
  for (const auto& stats : thread_stats) {
    failures += stats.failures;
    if (first_failure.empty() && !stats.first_failure.empty()) first_failure = stats.first_failure;
    for (std::size_t i = 0; i < strata_count; ++i) merge(merged[i], stats.strata[i]);
  }
  Aggregate total;
  for (const auto& value : merged) merge(total, value);
  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();

  std::ofstream csv(config.csv_path);
  csv << "mode,schedule,m,start_positions,n,k,D,cases,passed,valid,invalid,assertions,"
         "max_pair_cover_ratio,max_self_cover_ratio,tight_pair,tight_self\n";
  for (std::size_t mode = 0; mode < kModeNames.size(); ++mode) {
    for (std::size_t schedule = 0; schedule < kScheduleNames.size(); ++schedule) {
      for (std::size_t m_index = 0; m_index < kMValues.size(); ++m_index) {
        for (std::size_t n_index = 0; n_index < kStartCounts.size(); ++n_index) {
          const auto& value = merged[stratum_index(mode, schedule, m_index, n_index)];
          if (value.cases == 0) continue;
          const std::int64_t m = kMValues[m_index];
          const std::int64_t starts = kStartCounts[n_index];
          const std::int64_t n = starts == 0 ? std::max<std::int64_t>(0, m - 1)
                                             : starts + m - 1;
          const std::int64_t k = schedule_k(schedule, n);
          csv << kModeNames[mode] << ',' << kScheduleNames[schedule] << ',' << m << ','
              << starts << ',' << n << ',' << k << ',' << m + k << ',' << value.cases
              << ',' << value.passed << ',' << value.valid << ',' << value.invalid << ','
              << value.assertions << ',' << std::setprecision(12)
              << static_cast<double>(value.max_pair_ratio) << ','
              << static_cast<double>(value.max_self_ratio) << ',' << value.tight_pair << ','
              << value.tight_self << '\n';
        }
      }
    }
  }
  csv.close();

  std::ofstream summary(config.summary_path);
  summary << std::fixed << std::setprecision(6);
  summary << "{\n"
          << "  \"status\": \"" << (failures == 0 ? "PASS" : "FAIL") << "\",\n"
          << "  \"cases\": " << total.cases << ",\n"
          << "  \"passed\": " << total.passed << ",\n"
          << "  \"failures\": " << failures << ",\n"
          << "  \"valid_cases\": " << total.valid << ",\n"
          << "  \"invalid_cases_with_witness\": " << total.invalid << ",\n"
          << "  \"atomic_assertions\": " << total.assertions << ",\n"
          << "  \"threads\": " << config.threads << ",\n"
          << "  \"seed\": " << config.seed << ",\n"
          << "  \"m_min\": " << kMValues.front() << ",\n"
          << "  \"m_max\": " << kMValues.back() << ",\n"
          << "  \"maximum_start_positions\": " << kStartCounts.back() << ",\n"
          << "  \"schedule_count\": " << kScheduleNames.size() << ",\n"
          << "  \"case_mode_count\": " << kModeNames.size() << ",\n"
          << "  \"nonempty_strata\": "
          << std::count_if(merged.begin(), merged.end(),
                           [](const Aggregate& value) { return value.cases > 0; })
          << ",\n"
          << "  \"tight_pair_cases\": " << total.tight_pair << ",\n"
          << "  \"tight_self_cases\": " << total.tight_self << ",\n"
          << "  \"maximum_pair_cover_ratio\": "
          << static_cast<double>(total.max_pair_ratio) << ",\n"
          << "  \"maximum_self_cover_ratio\": "
          << static_cast<double>(total.max_self_ratio) << ",\n"
          << "  \"elapsed_seconds\": " << elapsed << ",\n"
          << "  \"first_failure\": \"" << first_failure << "\",\n"
          << "  \"strata_csv\": \""
          << std::filesystem::path(config.csv_path).filename().string() << "\",\n"
          << "  \"evidence_boundary\": \"Occurrence-position property tests support the geometric lemma but do not prove it, sample the full constrained language, or estimate capacity.\"\n"
          << "}\n";
  summary.close();

  std::cout << "status=" << (failures == 0 ? "PASS" : "FAIL")
            << " cases=" << total.cases << " assertions=" << total.assertions
            << " failures=" << failures << " elapsed_seconds=" << std::setprecision(4)
            << elapsed << '\n';
  if (!first_failure.empty()) std::cout << first_failure << '\n';
  return failures == 0 ? 0 : 1;
}
