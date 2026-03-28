#include <vonmises_generalized.hpp>
#include <gel.hpp>
#include <cstdio>
#include <algorithm>
#include <ranges>
#include <vector>
#include <cmath>

namespace vonmises_generalized {

namespace rg = std::ranges;
namespace vw = std::ranges::views;
using std::vector;

double calc_lambda(double f, double H, double G, double* Sigma_TR,
                   double* Sigma_0, double delta, double beta) {
    double G1 = G + H / 2;
    double A1 = f;
    double A2 = gel::Aij3x3norm(Sigma_TR) - gel::Aij3x3norm(Sigma_0);
    double A3 = delta - 2.0 * G;
    double A4 = (delta + H) * beta;
    double c1 = 2.0 * G1 * A3;
    double c2 = A4 - A1 * A3 + 2.0 * G1 * A2;
    double c3 = -A1 * A2;

    vector<double> vl{-(c2 - std::sqrt(c2 * c2 - 4.0 * c1 * c3)) / (2.0 * c1),
                      -(c2 + std::sqrt(c2 * c2 - 4.0 * c1 * c3)) / (2.0 * c1)};

    auto lambdas = vector<double>{
        std::from_range, vl | vw::filter([](auto l) { return l >= 0; })};
    if (lambdas.empty()) {
        printf("[ERROR] Couldn't find suitable lambdas.\n");
        exit(2);
    }

    double lambda_1 = *rg::min_element(lambdas);

    return lambda_1;
}

vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                       vonmises_generalized::ModelParams& mod_p,
                                       vector<double> hist) {
    // material params
    double young = mat_p.young;
    double poisson = mat_p.poisson;

    // vonmises params
    double sigma_y = mod_p.sigma_y * std::sqrt(2.0 / 3.0);
    double Hiso = mod_p.Hiso * (2.0 / 3.0);
    double Hkin = mod_p.Hkin * (2.0 / 3.0);
    double beta = mod_p.beta * std::sqrt(2.0 / 3.0);
    double delta = mod_p.delta * (2.0 / 3.0);

    // initial values
    double gamma_0 = 0.0;
    double s_0[9] = {0};
    double backstress_0[9] = {0};
    double epsilon_p_0[9] = {0};

    double G = young / (2.0 * (1.0 + poisson));
    double K = young / (3.0 * (1.0 - 2.0 * poisson));
    double Identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    vector<double> strains{0.0};
    vector<double> stresses{0.0};
    vector<std::tuple<double, double>> data;

    for (double epsilon_11 : hist) {
        double a{-poisson * epsilon_11};
        double epsilon_1[9] = {epsilon_11, 0, 0, 0, a, 0, 0, 0, a};

        // epsilon_e_TR = epsilon_1 - epsilon_p_0;
        double epsilon_e_TR[9];
        gel::Aij3x3_sub_Bij(epsilon_1, epsilon_p_0, epsilon_e_TR);

        double epsilon_e_TR_trace = gel::Aii3x3(epsilon_e_TR);

        double p_TR = K * epsilon_e_TR_trace;

        double sigma_TR[9];
        common::strain_to_stress(epsilon_e_TR, young, poisson, sigma_TR);

        double s_TR[9];
        common::calc_deviatoric_parcel(sigma_TR, s_TR);

        double Sigma_TR[9];
        gel::Aij3x3_sub_Bij(s_TR, backstress_0, Sigma_TR);

        double f = gel::Aij3x3norm(Sigma_TR) - (sigma_y + Hiso * gamma_0);

        if (f <= 0.0) {
            stresses.push_back(sigma_TR[0] - (sigma_TR[4] + sigma_TR[8]) / 2.0);
        } else {
            double H = Hiso + Hkin;

            // Sigma_0 = s_0 - backstress_0;
            double Sigma_0[9];
            gel::Aij3x3_sub_Bij(s_0, backstress_0, Sigma_0);

            double lambda_1 =
                calc_lambda(f, H, G, Sigma_TR, Sigma_0, delta, beta);

            double n[9];
            // Matrix3d n = common::calc_n(Sigma_TR);
            common::calc_n(Sigma_TR, n);

            // s_1 = s_TR - 2 * G * lambda_1 * n;
            double s_1[9];
            gel::Aij3x3_add_fBij(s_TR, -2 * G * lambda_1, n, s_1);

            // backstress_1 = backstress_0 + Hkin * lambda_1 * n;
            double backstress_1[9];
            gel::Aij3x3_add_fBij(backstress_0, Hkin * lambda_1, n,
                                 backstress_1);

            double gamma_1 = gamma_0 + lambda_1;

            // epsilon_e_1 = s_1 / (2.0 * G) + 1.0 / 3.0 * epsilon_e_TR_trace *
            // Identity;
            double epsilon_e_1[9];
            gel::eAij3x3_add_fBij(1.0 / (2.0 * G), s_1,
                                  1.0 / 3.0 * epsilon_e_TR_trace, Identity,
                                  epsilon_e_1);

            double epsilon_p_1[9];
            gel::Aij3x3_sub_Bij(epsilon_1, epsilon_e_1, epsilon_p_1);

            // update values
            gel::cpyarr(epsilon_p_1, 9, epsilon_p_0);
            gel::cpyarr(backstress_1, 9, backstress_0);
            gel::cpyarr(s_1, 9, s_0);
            gamma_0 = gamma_1;

            // sigma_1 = s_1 + p_TR * Identity;
            double sigma_1[9];
            gel::Aij3x3_add_fBij(s_1, p_TR, Identity, sigma_1);

            stresses.push_back(sigma_1[0] - (sigma_1[4] + sigma_1[8]) / 2.0);
        }

        strains.push_back(epsilon_1[0]);
    }

    for (auto [a, b] : vw::zip(strains, stresses)) {
        data.push_back({a, b});
    }
    return data;
}

}  // namespace vonmises_generalized
