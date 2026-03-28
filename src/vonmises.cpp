#include <vonmises.hpp>
#include <gel.hpp>
#include <ranges>
#include <vector>
#include <cmath>

namespace vonmises {

// namespace rg = std::ranges;
namespace vw = std::ranges::views;
using std::vector;

vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                       ModelParams& mod_p,
                                       vector<double> hist) {
    // material params
    double young = mat_p.young;
    double poisson = mat_p.poisson;

    // vonmises params
    double sigma_y = mod_p.sigma_y;
    double Hiso = mod_p.Hiso;
    double Hkin = mod_p.Hkin;

    // initial values
    double epsilon_barra_p_0 = 0.0;
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

        double e_e_TR[9];
        common::calc_deviatoric_parcel(epsilon_e_TR, e_e_TR);

        // s_TR = 2 * G * e_e_TR;
        double s_TR[9];
        gel::Aij3x3_mul_v(e_e_TR, 2.0 * G, s_TR);

        // eta_TR = s_TR - backstress_0;
        double eta_TR[9];
        gel::Aij3x3_sub_Bij(s_TR, backstress_0, eta_TR);

        double q_TR = std::sqrt(3.0 / 2.0) * gel::Aij3x3norm(eta_TR);

        double f = q_TR - (sigma_y + Hiso * epsilon_barra_p_0);

        if (f <= 0.0) {
            // sigma_1 = s_TR + p_TR * Identity;
            double sigma_1[9];
            gel::Aij3x3_add_fBij(s_TR, p_TR, Identity, sigma_1);

            stresses.push_back(sigma_1[0] - (sigma_1[4] + sigma_1[8]) / 2.0);
        } else {
            double gamma_1 = f / (3 * G + Hiso + Hkin);

            double epsilon_barra_p_1 = epsilon_barra_p_0 + gamma_1;

            double n[9];
            common::calc_n(eta_TR, n);

            // backstress_1 = backstress_0 + gamma_1 * sqrt(2.0 / 3.0) * Hkin *
            // n;
            double backstress_1[9];
            gel::Aij3x3_add_fBij(backstress_0,
                                 gamma_1 * std::sqrt(2.0 / 3.0) * Hkin, n,
                                 backstress_1);

            // s_1 = s_TR - 2 * G * gamma_1 * sqrt(3.0 / 2.0) * n;
            double s_1[9];
            gel::Aij3x3_add_fBij(
                s_TR, -2.0 * G * gamma_1 * std::sqrt(3.0 / 2.0), n, s_1);

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
            epsilon_barra_p_0 = epsilon_barra_p_1;

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

}  // namespace vonmises
