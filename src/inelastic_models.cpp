#include <gnuplot-iostream.h>

#include "mazars_model.hpp"
#include "mazars_mu_model.hpp"
#include "vonmises.hpp"
#include "vonmises_generalized.hpp"

void plot(auto& data, std::string titulo) {
    Gnuplot gp;
    gp << "set terminal qt noraise enhanced font 'Utopia,13'\n";
    gp << "unset key\n";
    gp << "set grid\n";
    gp << "set xlabel '𝜀'\n";
    gp << "set ylabel '𝜎'\n";
    gp << "set xtics font ',10'\n";
    gp << "set ytics font ',10'\n";
    gp << "set title '" << titulo << "'\n";
    gp << "plot '-' with lines lc 8 lw 2\n";
    gp.send1d(data);
    gp.flush();
}

int main() {
    auto vm = [] {
        common::MaterialProperties mat_p{100, 0.21};
        vonmises::ModelParams mod_p{0.4 * sqrt(2.0 / 3.0), 20, 0};

        auto hist = common::create_hist({{0, 0.04, 50}});
        auto res = vonmises::run(mat_p, mod_p, hist);
        plot(res, "Von Mises Model");
    };
    auto vm_generalized = [] {
        common::MaterialProperties mat_p{100, 0.21};
        vonmises_generalized::ModelParams mod_p{0.3 * sqrt(2.0 / 3.0), 20, 0,
                                                0.1 * sqrt(2.0 / 3.0), 20};

        auto hist = common::create_hist({{0, 0.04, 50}});
        auto res = vonmises_generalized::run(mat_p, mod_p, hist);

        plot(res, "Generalized Von Mises Model");
    };
    auto mazars_model = [] {
        common::MaterialProperties mat_p{3000, 0.21};
        mazars_model::ModelParams mod_p{0.5 * pow(7e-5, 2) + 7e-5, 0.995, 10000,
                                        0.85, 2000};

        auto hist_traction = common::create_hist({{0, 0.0006, 100}});
        auto res_traction = mazars_model::run(mat_p, mod_p, hist_traction);
        auto hist_compression = common::create_hist({{0, -0.008, 100}});
        auto res_compression =
            mazars_model::run(mat_p, mod_p, hist_compression);

        Gnuplot gp;
        gp << "set terminal qt noraise enhanced font 'Utopia,13'\n";
        gp << "unset key\n";
        gp << "set grid\n";
        gp << "set xlabel '𝜀'\n";
        gp << "set ylabel '𝜎'\n";
        gp << "set xtics font ',10'\n";
        gp << "set ytics font ',10'\n";
        gp << "set title 'Mazars Model'\n";
        gp << "plot '-' with lines lc 8 lw 2, '-' with lines lc 8 lw 2\n";
        gp.send1d(res_traction);
        gp.send1d(res_compression);
        gp.flush();
    };
    auto mu_model = [] {
        common::MaterialProperties mat_p{3000, 0.21};
        mazars_mu_model::ModelParams mod_p{0.5 * pow(7e-5, 2) + 7e-5,
                                           0.5 * pow(3e-4, 2) + 3e-4,
                                           0.995,
                                           15000,
                                           0.85,
                                           1620,
                                           1.0};

        auto hist = common::create_hist({{0, 0.00015, 100},
                                         {0.00015, 0, 100},
                                         {0, -0.001, 100},
                                         {-0.001, 0, 100},
                                         {0, 0.0002, 100},
                                         {0.0002, 0, 100},
                                         {0, -0.003, 100},
                                         {-0.003, 0, 100},
                                         {0, 0.0003, 100},
                                         {0.0003, 0, 100},
                                         {0, -0.005, 100}});
        auto res = mazars_mu_model::run(mat_p, mod_p, hist);

        plot(res, "Mazars μ-Model");
    };

    vm();
    vm_generalized();
    mazars_model();
    mu_model();
};
