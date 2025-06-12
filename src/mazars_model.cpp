#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <print>
#include <ranges>
#include <vector>

#include "common.hpp"

namespace mazars_model {

namespace rg = std::ranges;
namespace vw = std::ranges::views;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::println;
using std::vector;

Vector3d calc_valores_positivos(Vector3d& valores_principais) {
    return valores_principais.unaryExpr(
        [](double v) { return v >= 0.0 ? v : 0.0; });
}

double calc_epsilon_equiv(Vector3d& epsilon_pos) {
    double epsilon_equiv{1e-14};
    for (auto eps : epsilon_pos) {
        epsilon_equiv += eps * eps;
    }
    return sqrt(epsilon_equiv);
}

double calc_alpha_t(Vector3d& epsilon_t, Vector3d& epsilon_pos,
                    double epsilon_equiv) {
    double epsilon_equiv_2 = pow(epsilon_equiv, 2);
    double alpha_t{0};
    for (auto [eps, eps_pos] : vw::zip(epsilon_t, epsilon_pos)) {
        alpha_t += eps * eps_pos / epsilon_equiv_2;
    }
    return alpha_t;
}

double calc_Di(double epsilon_equiv, double epsilon_d0, double Ai, double Bi) {
    return 1.0 - epsilon_d0 * (1.0 - Ai) / epsilon_equiv -
           Ai * std::pow(std::numbers::e, -Bi * (epsilon_equiv - epsilon_d0));
}

struct ModelParams {
    double epsilon_d0;
    double At;
    double Bt;
    double Ac;
    double Bc;

    ModelParams(double epsilon_d0, double At, double Bt, double Ac, double Bc)
        : epsilon_d0{epsilon_d0}, At{At}, Bt{Bt}, Ac{Ac}, Bc{Bc} {};
};

std::tuple<double, double> calc_D_kappa(Matrix3d& sigma, double young,
                                        double poisson, double dano,
                                        double kappa, ModelParams& mp) {
    auto sigma_princ = common::calc_valores_principais(sigma);
    auto sigma_pos = calc_valores_positivos(sigma_princ);

    auto epsilon_princ =
        common::sigma_princ_to_epsilon_princ(sigma_princ, young, poisson);
    auto epsilon_pos = calc_valores_positivos(epsilon_princ);
    double epsilon_equiv = calc_epsilon_equiv(epsilon_pos);

    if (epsilon_equiv > kappa) {
        Vector3d epsilon_t =
            common::sigma_princ_to_epsilon_princ(sigma_pos, young, poisson);
        double alpha_t = calc_alpha_t(epsilon_t, epsilon_pos, epsilon_equiv);
        double alpha_c = 1 - alpha_t;
        double D_t = calc_Di(epsilon_equiv, mp.epsilon_d0, mp.At, mp.Bt);
        double D_c = calc_Di(epsilon_equiv, mp.epsilon_d0, mp.Ac, mp.Bc);
        dano = alpha_t * D_t + alpha_c * D_c;
        kappa = epsilon_equiv;
    }
    return {dano, kappa};
}

vector<std::tuple<double, double>> run(vector<double> hist) {
    // material params
    double poisson = 0.21;
    double young0 = 3000;

    // mu model params
    ModelParams mp{0.5 * pow(7e-5, 2) + 7e-5, 0.995, 10000, 0.85, 2000};

    // initial values
    double kappa_0 = mp.epsilon_d0;
    double dano_0 = 0;

    vector<double> strains{0.0};
    vector<double> stresses{0.0};
    vector<std::tuple<double, double>> data;

    for (double epsilon_11 : hist) {
        Matrix3d epsilon_1{
            {epsilon_11, 0, 0},
            {0, -poisson * epsilon_11, 0},
            {0, 0, -poisson * epsilon_11},
        };
        double young = (1 - dano_0) * young0;
        Matrix3d sigma_TR = common::strain_to_stress(epsilon_1, young, poisson);
        auto [dano_1, kappa_1] =
            calc_D_kappa(sigma_TR, young, poisson, dano_0, kappa_0, mp);
        young = (1 - dano_1) * young0;
        Matrix3d sigma_1 = common::strain_to_stress(epsilon_1, young, poisson);

        kappa_0 = kappa_1;
        dano_0 = dano_1;

        strains.push_back(epsilon_1(0, 0));
        stresses.push_back(sigma_1(0, 0));
    }

    for (auto [a, b] : vw::zip(strains, stresses)) {
        data.push_back({a, b});
    }
    return data;
}
}  // namespace mazars_model
