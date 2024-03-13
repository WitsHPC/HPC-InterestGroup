#include <any>
#include <chrono>
#include <functional>

// For typesafe code.
typedef std::function<std::any()> single_func;
typedef bool (*validator_func)(std::any, std::any);

bool double_comparison(std::any a, std::any b) {
    return abs(std::any_cast<double>(a) - std::any_cast<double>(b)) < 1e-5;
}

std::pair<double, double> calc_mean_std(const std::vector<double> &v){
    double mean = 0.0;
    for (auto &k: v){
        mean += k;
    }
    mean /= v.size();
    // std dev
    double var = 0.0;
    for (auto &k: v){
        var += pow(k - mean, 2);
    }
    var /= v.size();
    return {mean, sqrt(var)};
}

/**
 * @brief A class that times a few functions, validates them and displays it properly.
 * 
 */
class Timer {
    std::vector<single_func> functions;
    std::vector<std::string> names;
    validator_func function_to_validate;
    std::any correct_answer;
    int num_exps_to_run;

   public:
   /**
    * @brief Construct a new Timer object
    * 
    * @param _functions List of funtions
    * @param _names List of names, same length as the above.
    * @param _function_to_validate How do we validate if the answers are correct.
    * @param _correct_answer The correct answer to validate against.
    * @param _num_exps_to_run How many runs should we average across.
    */
    Timer(
        std::vector<single_func> _functions,
        std::vector<std::string> _names,
        validator_func _function_to_validate,
        std::any _correct_answer,
        int _num_exps_to_run = 100) : functions(_functions),
                                      names(_names),
                                      function_to_validate(_function_to_validate),
                                      num_exps_to_run(_num_exps_to_run),
                                      correct_answer(_correct_answer) {}

    void run() {
        std::vector<std::vector<double>> times(functions.size(), std::vector<double>(num_exps_to_run, 0.0));
        std::vector<bool> is_corrects(functions.size(), true);

        for (int f = 0; f < functions.size(); ++f) {
            // repeat X times to average out noise.
            for (int i = 0; i < num_exps_to_run; ++i) {
                // get the time now
                auto start = std::chrono::high_resolution_clock::now();
                // perform the result
                std::any ans = functions[f]();
                // end time
                auto end = std::chrono::high_resolution_clock::now();
                // duration
                auto duration = std::chrono::duration<double, std::milli>(end - start).count();
                times[f][i] = duration;

                bool x = function_to_validate(correct_answer, ans);
                is_corrects[f] = is_corrects[f] && x;
            }
        }

        // printf magic.
        printf("| %-25s | %-15s | %-15s | %-7s |\n", "Method", "Correct Answer", "Time (ms)", "Speedup");
        printf("%s\n",std::string(25 + 15 + 15 + 8 + 13,'-').c_str());
        for (int f = 0; f < functions.size(); ++f) {
            auto [mean, std] = calc_mean_std(times[f]);
            times[f][0] = mean;
            bool is_correct = is_corrects[f];
            printf("| %-25s | %-15s | %-8.2lf ± %4.1lf | %-7.2lf |\n", names[f].c_str(), is_correct ? "YES" : "NO", mean, std, times[0][0] / times[f][0]);
        }
    }
};
