// Exhaustive realizable-word audit for the stem reduction and orbit cover.
//
// Unlike orbit_interval_million_audit.cpp, this program enumerates actual
// words.  It computes the original all-stem rule and the length-m reduction
// independently, then checks the orbit cover for every avoiding (m,k) cell.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Job {
  const char* name;
  int q;
  int n;
  std::array<int, 4> theta;
};

constexpr std::array<Job, 5> kFullJobs = {{
    {"binary_swap", 2, 20, {1, 0, 0, 0}},
    {"binary_identity", 2, 20, {0, 1, 0, 0}},
    {"ternary_fixed_point", 3, 12, {1, 0, 2, 0}},
    {"dna_watson_crick", 4, 10, {3, 2, 1, 0}},
    {"quaternary_identity", 4, 10, {0, 1, 2, 3}},
}};

constexpr std::array<Job, 5> kQuickJobs = {{
    {"binary_swap", 2, 12, {1, 0, 0, 0}},
    {"binary_identity", 2, 12, {0, 1, 0, 0}},
    {"ternary_fixed_point", 3, 8, {1, 0, 2, 0}},
    {"dna_watson_crick", 4, 7, {3, 2, 1, 0}},
    {"quaternary_identity", 4, 7, {0, 1, 2, 3}},
}};

// This profile is intentionally expensive: it increases both word length and
// the audited stem range.  It is kept separate from the normal regression run
// so that routine tests remain fast and deterministic.
constexpr std::array<Job, 5> kExtendedJobs = {{
    {"binary_swap_extended", 2, 24, {1, 0, 0, 0}},
    {"binary_identity_extended", 2, 24, {0, 1, 0, 0}},
    {"ternary_fixed_point_extended", 3, 14, {1, 0, 2, 0}},
    {"dna_watson_crick_extended", 4, 12, {3, 2, 1, 0}},
    {"quaternary_identity_extended", 4, 12, {0, 1, 2, 3}},
}};

struct Stats {
  std::uint64_t words = 0;
  std::uint64_t stem_cells = 0;
  std::uint64_t word_parameter_cells = 0;
  std::uint64_t avoiding_parameter_cells = 0;
  std::uint64_t orbit_cover_cells = 0;
  std::uint64_t exact_gap_equalities = 0;
  std::uint64_t atomic_assertions = 0;
  std::uint64_t failures = 0;
  std::uint64_t digest_xor = 0;
  std::string first_failure;
};

struct Config {
  unsigned threads = 1;
  bool quick = false;
  bool extended = false;
  std::string summary_path = "output/full_word_definition_audit_summary.json";
  std::string csv_path = "output/full_word_definition_audit_jobs.csv";
};

std::uint64_t splitmix_hash(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t integer_power(int base, int exponent) {
  std::uint64_t result = 1;
  for (int i = 0; i < exponent; ++i) result *= static_cast<std::uint64_t>(base);
  return result;
}

void merge(Stats& target, const Stats& source) {
  target.words += source.words;
  target.stem_cells += source.stem_cells;
  target.word_parameter_cells += source.word_parameter_cells;
  target.avoiding_parameter_cells += source.avoiding_parameter_cells;
  target.orbit_cover_cells += source.orbit_cover_cells;
  target.exact_gap_equalities += source.exact_gap_equalities;
  target.atomic_assertions += source.atomic_assertions;
  target.failures += source.failures;
  target.digest_xor ^= source.digest_xor;
  if (target.first_failure.empty() && !source.first_failure.empty()) {
    target.first_failure = source.first_failure;
  }
}

Config parse_args(int argc, char** argv) {
  Config config;
  const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
  config.threads = std::max(1U, hardware > 2 ? hardware - 2 : hardware);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) {
        std::cerr << "missing argument value\n";
        std::exit(2);
      }
      return argv[i];
    };
    if (arg == "--threads") config.threads = std::stoul(value());
    else if (arg == "--summary") config.summary_path = value();
    else if (arg == "--csv") config.csv_path = value();
    else if (arg == "--quick") config.quick = true;
    else if (arg == "--extended") config.extended = true;
    else {
      std::cerr << "unknown argument: " << arg << '\n';
      std::exit(2);
    }
  }
  return config;
}

std::uint64_t reverse_complement_code(std::uint64_t code, int m, const Job& job) {
  std::uint64_t result = 0;
  for (int i = 0; i < m; ++i) {
    const int symbol = static_cast<int>(code % static_cast<std::uint64_t>(job.q));
    code /= static_cast<std::uint64_t>(job.q);
    result = result * static_cast<std::uint64_t>(job.q) +
             static_cast<std::uint64_t>(job.theta[symbol]);
  }
  return result;
}

bool verify_orbit_cover(const Job& job, int m, int k,
                        const std::vector<std::vector<std::uint64_t>>& forward,
                        Stats& stats, std::string& failure) {
  const int starts = job.n - m + 1;
  const int distance = m + k;
  // Only at most `starts` blocks occur in a word.  Storing just those blocks
  // avoids allocating q^m empty vectors and makes all m<=n practical in the
  // extended exhaustive profile.
  std::vector<std::uint64_t> touched;
  std::vector<std::vector<int>> positions;
  touched.reserve(static_cast<std::size_t>(starts));
  positions.reserve(static_cast<std::size_t>(starts));
  for (int start = 0; start < starts; ++start) {
    const auto block = forward[m][start];
    const auto found = std::find(touched.begin(), touched.end(), block);
    if (found == touched.end()) {
      touched.push_back(block);
      positions.push_back({start});
    } else {
      positions[static_cast<std::size_t>(found - touched.begin())].push_back(start);
    }
  }
  std::vector<unsigned char> processed(touched.size(), 0);
  for (std::size_t block_index = 0; block_index < touched.size(); ++block_index) {
    const auto block = touched[block_index];
    if (processed[block_index]) continue;
    const auto partner = reverse_complement_code(block, m, job);
    processed[block_index] = 1;
    const auto partner_it = std::find(touched.begin(), touched.end(), partner);
    static const std::vector<int> empty_positions;
    const auto& left = positions[block_index];
    const auto& right =
        partner_it == touched.end()
            ? empty_positions
            : positions[static_cast<std::size_t>(partner_it - touched.begin())];
    if (partner_it != touched.end()) {
      processed[static_cast<std::size_t>(partner_it - touched.begin())] = 1;
    }
    if (block == partner) {
      if (left.size() >= 2) {
        ++stats.atomic_assertions;
        if (left.back() - left.front() >= distance) {
          failure = "self-RC orbit span is not strictly below D";
          return false;
        }
      }
      continue;
    }

    const std::vector<int>* selected = nullptr;
    const std::vector<int>* noncore = nullptr;
    if (!left.empty() && right.empty()) {
      selected = &left;
      noncore = &right;
    } else if (left.empty() && !right.empty()) {
      selected = &right;
      noncore = &left;
    } else if (block < partner) {
      selected = &left;
      noncore = &right;
    } else {
      selected = &right;
      noncore = &left;
    }
    if (noncore->empty()) continue;
    ++stats.atomic_assertions;
    if (selected->empty()) {
      failure = "noncore occurs without a selected anchor";
      return false;
    }
    const int anchor = selected->front();
    const int lo = anchor - distance + 1;
    const int hi = anchor + distance - 1;
    for (const int start : *noncore) {
      ++stats.atomic_assertions;
      if (start < lo || start > hi) {
        failure = "paired orbit is not covered by the anchor interval";
        return false;
      }
    }
    ++stats.atomic_assertions;
    if (noncore->back() - noncore->front() + 1 > 2 * distance - 1) {
      failure = "paired orbit cover exceeds 2D-1";
      return false;
    }
  }
  return true;
}

bool audit_word(std::uint64_t word_index, const Job& job, int maximum_m, Stats& stats,
                std::string& failure) {
  std::vector<int> word(static_cast<std::size_t>(job.n));
  std::uint64_t encoded = word_index;
  for (int i = job.n - 1; i >= 0; --i) {
    word[static_cast<std::size_t>(i)] =
        static_cast<int>(encoded % static_cast<std::uint64_t>(job.q));
    encoded /= static_cast<std::uint64_t>(job.q);
  }

  std::vector<std::uint64_t> powers(static_cast<std::size_t>(job.n + 1), 1);
  for (int i = 1; i <= job.n; ++i) {
    powers[i] = powers[i - 1] * static_cast<std::uint64_t>(job.q);
  }
  std::vector<std::vector<std::uint64_t>> forward(static_cast<std::size_t>(job.n + 1));
  std::vector<std::vector<std::uint64_t>> reverse(static_cast<std::size_t>(job.n + 1));
  for (int length = 1; length <= job.n; ++length) {
    forward[length].resize(static_cast<std::size_t>(job.n - length + 1));
    reverse[length].resize(static_cast<std::size_t>(job.n - length + 1));
  }
  for (int start = 0; start < job.n; ++start) {
    std::uint64_t f = 0;
    std::uint64_t r = 0;
    for (int length = 1; start + length <= job.n; ++length) {
      const int symbol = word[static_cast<std::size_t>(start + length - 1)];
      f = f * static_cast<std::uint64_t>(job.q) + static_cast<std::uint64_t>(symbol);
      r += static_cast<std::uint64_t>(job.theta[symbol]) * powers[length - 1];
      forward[length][start] = f;
      reverse[length][start] = r;
    }
  }

  std::vector<int> exact_max_gap(static_cast<std::size_t>(job.n + 2), -1);
  for (int length = 1; length <= job.n; ++length) {
    const int starts = job.n - length + 1;
    for (int left = 0; left < starts; ++left) {
      for (int right = left + length; right < starts; ++right) {
        ++stats.atomic_assertions;
        if (forward[length][left] == reverse[length][right]) {
          exact_max_gap[length] = std::max(exact_max_gap[length], right - left - length);
        }
      }
    }
  }
  std::vector<int> full_max_gap(static_cast<std::size_t>(job.n + 2), -1);
  for (int m = job.n; m >= 1; --m) {
    full_max_gap[m] = std::max(exact_max_gap[m], full_max_gap[m + 1]);
  }

  maximum_m = std::min(maximum_m, job.n);
  for (int m = 1; m <= maximum_m; ++m) {
    ++stats.stem_cells;
    ++stats.atomic_assertions;
    if (full_max_gap[m] != exact_max_gap[m]) {
      failure = "all-stem maximum gap differs from length-m maximum gap";
      return false;
    }
    ++stats.exact_gap_equalities;
    for (int k = 0; k <= job.n; ++k) {
      ++stats.word_parameter_cells;
      const bool original = full_max_gap[m] < k;
      const bool reduced = exact_max_gap[m] < k;
      ++stats.atomic_assertions;
      if (original != reduced) {
        failure = "original and reduced definitions disagree";
        return false;
      }
      if (!reduced) continue;
      ++stats.avoiding_parameter_cells;
      if (!verify_orbit_cover(job, m, k, forward, stats, failure)) return false;
      ++stats.orbit_cover_cells;
      stats.digest_xor ^= splitmix_hash(word_index ^
                                        (static_cast<std::uint64_t>(m) << 48U) ^
                                        (static_cast<std::uint64_t>(k) << 56U));
    }
  }
  ++stats.words;
  return true;
}

Stats run_job(const Job& job, unsigned threads, int maximum_m) {
  const std::uint64_t word_count = integer_power(job.q, job.n);
  std::vector<Stats> local(threads);
  std::vector<std::thread> workers;
  std::atomic<bool> stop{false};
  workers.reserve(threads);
  for (unsigned thread_id = 0; thread_id < threads; ++thread_id) {
    workers.emplace_back([&, thread_id]() {
      for (std::uint64_t index = thread_id; index < word_count && !stop.load();
           index += threads) {
        std::string failure;
        if (!audit_word(index, job, maximum_m, local[thread_id], failure)) {
          ++local[thread_id].failures;
          local[thread_id].first_failure =
              "job=" + std::string(job.name) + " word=" + std::to_string(index) + " " + failure;
          stop.store(true);
          break;
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
  Stats result;
  for (const auto& value : local) merge(result, value);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const Config config = parse_args(argc, argv);
  if (config.quick && config.extended) {
    std::cerr << "--quick and --extended are mutually exclusive\n";
    return 2;
  }
  const auto& jobs = config.quick ? kQuickJobs : (config.extended ? kExtendedJobs : kFullJobs);
  const int maximum_m = config.extended ? 24 : 6;
  std::vector<Stats> results;
  results.reserve(jobs.size());
  Stats total;
  const auto all_start = std::chrono::steady_clock::now();
  for (const auto& job : jobs) {
    const auto start = std::chrono::steady_clock::now();
    Stats result = run_job(job, config.threads, maximum_m);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "job=" << job.name << " words=" << result.words
              << " parameter_cells=" << result.word_parameter_cells
              << " failures=" << result.failures << " elapsed_seconds=" << std::fixed
              << std::setprecision(3) << elapsed << '\n';
    results.push_back(result);
    merge(total, result);
    if (result.failures != 0) break;
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - all_start).count();

  std::ofstream csv(config.csv_path);
  csv << "job,q,n,words,stem_cells,word_parameter_cells,avoiding_parameter_cells,"
         "orbit_cover_cells,exact_gap_equalities,atomic_assertions,failures,digest_xor\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& job = jobs[i];
    const auto& value = results[i];
    csv << job.name << ',' << job.q << ',' << job.n << ',' << value.words << ','
        << value.stem_cells << ',' << value.word_parameter_cells << ','
        << value.avoiding_parameter_cells << ',' << value.orbit_cover_cells << ','
        << value.exact_gap_equalities << ',' << value.atomic_assertions << ','
        << value.failures << ',' << value.digest_xor << '\n';
  }
  csv.close();

  std::ofstream summary(config.summary_path);
  summary << "{\n"
          << "  \"status\": \"" << (total.failures == 0 ? "PASS" : "FAIL") << "\",\n"
          << "  \"mode\": \""
          << (config.quick ? "quick" : (config.extended ? "extended" : "full"))
          << "\",\n"
          << "  \"jobs\": " << results.size() << ",\n"
          << "  \"threads\": " << config.threads << ",\n"
          << "  \"words\": " << total.words << ",\n"
          << "  \"stem_cells\": " << total.stem_cells << ",\n"
          << "  \"word_parameter_cells\": " << total.word_parameter_cells << ",\n"
          << "  \"avoiding_parameter_cells\": " << total.avoiding_parameter_cells << ",\n"
          << "  \"orbit_cover_cells\": " << total.orbit_cover_cells << ",\n"
          << "  \"exact_gap_equalities\": " << total.exact_gap_equalities << ",\n"
          << "  \"atomic_assertions\": " << total.atomic_assertions << ",\n"
          << "  \"failures\": " << total.failures << ",\n"
          << "  \"digest_xor\": " << total.digest_xor << ",\n"
          << "  \"elapsed_seconds\": " << std::fixed << std::setprecision(6) << elapsed
          << ",\n"
          << "  \"first_failure\": \"" << total.first_failure << "\",\n"
          << "  \"evidence_boundary\": \"Exhaustive finite-word agreement can expose counterexamples but does not replace the universal proof.\"\n"
          << "}\n";
  summary.close();
  return total.failures == 0 ? 0 : 1;
}
