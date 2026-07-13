#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int reverse_complement_dinucleotide(int word) {
    const int left = word / 4;
    const int right = word % 4;
    return (3 - right) * 4 + (3 - left);
}

struct Row {
    int r;
    std::size_t states;
    long double log2_count;
};

std::vector<Row> count_pairs(int max_r, const std::unordered_set<int>& checkpoints) {
    using Map = std::unordered_map<std::uint32_t, long double>;
    Map current;
    Map next;
    current.reserve(16);
    for (int left = 0; left < 4; ++left) {
        for (int right = 0; right < 4; ++right) {
            const auto key = static_cast<std::uint32_t>((left << 2) | right);
            current[key] += 1.0L;
        }
    }

    std::vector<Row> rows;
    if (checkpoints.contains(0)) {
        rows.push_back({0, current.size(), 4.0L});
    }

    for (int r = 1; r <= max_r; ++r) {
        next.clear();
        next.reserve(std::min<std::size_t>(600000, std::max<std::size_t>(64, current.size() * 3)));
        next.max_load_factor(0.72F);

        for (const auto& [key, multiplicity] : current) {
            const int right_suffix = static_cast<int>(key & 3U);
            const int left_suffix = static_cast<int>((key >> 2U) & 3U);
            const std::uint32_t seen = key >> 4U;
            for (int left_symbol = 0; left_symbol < 4; ++left_symbol) {
                const int left_word = left_suffix * 4 + left_symbol;
                const std::uint32_t updated_seen = seen | (1U << left_word);
                for (int right_symbol = 0; right_symbol < 4; ++right_symbol) {
                    const int right_word = right_suffix * 4 + right_symbol;
                    if (updated_seen & (1U << reverse_complement_dinucleotide(right_word))) {
                        continue;
                    }
                    const auto next_key =
                        (updated_seen << 4U) | static_cast<std::uint32_t>((left_symbol << 2) | right_symbol);
                    next[next_key] += multiplicity;
                }
            }
        }
        current.swap(next);

        if (r % 500 == 0 || r == max_r) {
            std::cerr << "progress r=" << r << " states=" << current.size() << '\n';
        }

        if (checkpoints.contains(r)) {
            long double total = 0.0L;
            for (const auto& [key, multiplicity] : current) {
                static_cast<void>(key);
                total += multiplicity;
            }
            rows.push_back({r, current.size(), std::log2(total)});
        }
    }
    return rows;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: paired_count_dna_m2 OUTPUT.csv [MAX_R]\n";
        return 2;
    }
    const int max_r = argc == 3 ? std::stoi(argv[2]) : 200;
    if (max_r < 0 || max_r > 5000) {
        throw std::invalid_argument("MAX_R must be between 0 and 5000");
    }
    std::vector<int> requested = {
        0, 1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200,
    };
    for (int r = 250; r <= max_r; r += 50) {
        requested.push_back(r);
    }
    requested.erase(
        std::remove_if(requested.begin(), requested.end(),
                       [max_r](int value) { return value > max_r; }),
        requested.end());
    if (requested.empty() || requested.back() != max_r) {
        requested.push_back(max_r);
    }
    const std::unordered_set<int> checkpoints(requested.begin(), requested.end());
    const auto rows = count_pairs(max_r, checkpoints);

    std::ofstream output(argv[1]);
    if (!output) {
        throw std::runtime_error("cannot open output file");
    }
    const long double lambda = 3.0L * (1.0L + std::sqrt(13.0L)) / 2.0L;
    const long double log2_lambda = std::log2(lambda);
    output << "r,states,log2_F,log2_F_over_r,log2_lambda,normalized_excess\n";
    output << std::setprecision(18);
    for (const auto& row : rows) {
        const long double normalized = row.r == 0 ? 0.0L : row.log2_count / row.r;
        const long double excess = row.r == 0 ? 0.0L : normalized - log2_lambda;
        output << row.r << ',' << row.states << ',' << row.log2_count << ',' << normalized << ','
               << log2_lambda << ',' << excess << '\n';
    }
    std::cout << "PAIRED_DNA_M2_CPP_PASS rows=" << rows.size() << " max_r=" << max_r
              << " final_states=" << rows.back().states << " final_excess="
              << static_cast<double>(rows.back().log2_count / rows.back().r - log2_lambda) << '\n';
    return 0;
}
