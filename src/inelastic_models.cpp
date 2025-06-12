#include <gnuplot-iostream.h>

#include <print>

#include "common.hpp"

namespace vonmises {
std::vector<std::tuple<double, double>> run(std::vector<double> hist);
}  // namespace vonmises

namespace vonmises_generalized {
std::vector<std::tuple<double, double>> run(std::vector<double> hist);
}  // namespace vonmises_generalized

namespace mazars_model {
std::vector<std::tuple<double, double>> run(std::vector<double> hist);
}  // namespace mazars_model

namespace mazars_mu_model {
std::vector<std::tuple<double, double>> run(std::vector<double> hist);
}  // namespace mazars_mu_model

int main() {
    auto vm = [] {
        auto hist = common::create_hist({{0, 0.04, 50}});
        auto res = vonmises::run(hist);

        Gnuplot gp;
        gp << "unset key\n";
        gp << "set grid\n";
        gp << "plot '-' with lines lc 8 lw 2\n";
        gp.send1d(res);
        gp.flush();
    };
    auto vm_generalized = [] {
        auto hist = common::create_hist({{0, 0.04, 50}});
        auto res = vonmises_generalized::run(hist);

        Gnuplot gp;
        gp << "unset key\n";
        gp << "set grid\n";
        gp << "plot '-' with lines lc 8 lw 2\n";
        gp.send1d(res);
        gp.flush();
    };
    auto mazars_model = [] {
        auto hist_traction = common::create_hist({{0, 0.0006, 100}});
        auto res_traction = mazars_model::run(hist_traction);
        auto hist_compression = common::create_hist({{0, -0.008, 100}});
        auto res_compression = mazars_model::run(hist_compression);

        Gnuplot gp;
        gp << "unset key\n";
        gp << "set grid\n";
        gp << "plot '-' with lines lc 8 lw 2, '-' with lines lc 8 lw 2\n";
        gp.send1d(res_traction);
        gp.send1d(res_compression);
        gp.flush();
    };
    auto mu_model = [] {
        auto hist = common::create_hist({{0, 0.00015, 100},
                                         {0.00015, 0, 100},
                                         {0, -0.002, 100},
                                         {-0.002, 0, 100},
                                         {0, 0.0002, 100},
                                         {0.0002, 0, 100},
                                         {0, -0.004, 100},
                                         {-0.004, 0, 100},
                                         {0, 0.0003, 100},
                                         {0.0003, 0, 100},
                                         {0, -0.006, 100},
                                         {-0.006, 0, 100},
                                         {0, 0.0006, 100},
                                         {0.0006, 0, 100}});
        auto res = mazars_mu_model::run(hist);

        Gnuplot gp;
        gp << "unset key\n";
        gp << "set grid\n";
        gp << "plot '-' with lines lc 8 lw 2\n";
        gp.send1d(res);
        gp.flush();
    };

    vm();
    vm_generalized();
    mazars_model();
    mu_model();
};
