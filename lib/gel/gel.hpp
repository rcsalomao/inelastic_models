namespace gel {

void printAij3x3(double const* A);
void printai(double const* a, unsigned int n);
void cpyarr(double const* a, unsigned int n, double* b);

void ai2_add_v(double const* a, double v, double* b);
void ai3_add_v(double const* a, double v, double* b);
void ai4_add_v(double const* a, double v, double* b);
void ai_add_v(double const* a, unsigned int n, double v, double* b);

void ai2_mul_v(double const* a, double v, double* b);
void ai3_mul_v(double const* a, double v, double* b);
void ai4_mul_v(double const* a, double v, double* b);
void ai_mul_v(double const* a, unsigned int n, double v, double* b);

void Aij3x3_add_v(double const* A, double v, double* B);
void Aij3x3_mul_v(double const* A, double v, double* B);

void ai2_add_bi(double const* a, double const* b, double* c);
void ai3_add_bi(double const* a, double const* b, double* c);
void ai4_add_bi(double const* a, double const* b, double* c);
void ai_add_bi(double const* a, double const* b, unsigned int n, double* c);

void ai2_add_ebi(double const* a, double e, double const* b, double* c);
void ai3_add_ebi(double const* a, double e, double const* b, double* c);
void ai4_add_ebi(double const* a, double e, double const* b, double* c);
void ai_add_ebi(double const* a, double e, double const* b, unsigned int n,
                double* c);

void ai2_sub_bi(double const* a, double const* b, double* c);
void ai3_sub_bi(double const* a, double const* b, double* c);
void ai4_sub_bi(double const* a, double const* b, double* c);
void ai_sub_bi(double const* a, double const* b, unsigned int n, double* c);

void ai2_mul_bi(double const* a, double const* b, double* c);
void ai3_mul_bi(double const* a, double const* b, double* c);
void ai4_mul_bi(double const* a, double const* b, double* c);
void ai_mul_bi(double const* a, double const* b, unsigned int n, double* c);

void ai2_div_bi(double const* a, double const* b, double* c);
void ai3_div_bi(double const* a, double const* b, double* c);
void ai4_div_bi(double const* a, double const* b, double* c);
void ai_div_bi(double const* a, double const* b, unsigned int n, double* c);

void Aij3x3_add_Bij(double const* A, double const* B, double* C);
void Aij3x3_sub_Bij(double const* A, double const* B, double* C);
void Aij3x3_mul_Bij(double const* A, double const* B, double* C);
void Aij3x3_div_Bij(double const* A, double const* B, double* C);
void Aij3x3_add_fBij(double const* A, double f, double const* B, double* C);
void eAij3x3_add_fBij(double e, double const* A, double f, double const* B,
                      double* C);
void eAij3x3_add_fBij_add_gCij(double e, double const* A, double f,
                               double const* B, double g, double* C, double* D);

double ai2bi(double const* a, double const* b);
double ai3bi(double const* a, double const* b);
double ai4bi(double const* a, double const* b);
double aibi(double const* a, double const* b, unsigned int n);

void Aij3x3Bjk(double const* A, double const* B, double* C);
void Aji3x3Bjk(double const* A, double const* B, double* C);
void Aij3x3Bkj(double const* A, double const* B, double* C);
void Aji3x3Bkj(double const* A, double const* B, double* C);

void Aij3x3bj(double const* A, double const* b, double* c);
void Aji3x3bj(double const* A, double const* b, double* c);

void ai3bjcross(double const* a, double const* b, double* c);
void ai3bj(double const* a, double const* b, double* C);

double ai3norm(double const* a);
void ai3unit(double const* a, double* b);
void Aij3x3trans(double* A);
void Aij3x3trans(double const* A, double* B);
double Aij3x3det(double const* A);
double Aij3x3norm(double const* A);
void Aij3x3inv(double const* A, double* B);
void Rji3x3AjkRkl(double const* R, double const* A, double* B);
void Aij3x3eigen_sym(double const* A, double* e_val, double* E_vec);

double Aii2x2(double const* A);
double Aii3x3(double const* A);

}  // namespace gel
