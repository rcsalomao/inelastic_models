#include <gel.hpp>
#include <immintrin.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

namespace gel {

void printAij3x3(double const* A) {
    printf("%10.3f %10.3f %10.3f\n", A[0], A[1], A[2]);
    printf("%10.3f %10.3f %10.3f\n", A[3], A[4], A[5]);
    printf("%10.3f %10.3f %10.3f\n", A[6], A[7], A[8]);
}

void printai(double const* a, unsigned int n) {
    for (size_t i{0}; i < n; i++) {
        printf("%.3f ", a[i]);
    }
    printf("\n");
}

void cpyarr(double const* a, unsigned int n, double* b) {
    for (size_t i{0}; i < n; i++) {
        b[i] = a[i];
    }
}

void ai2_add_v(double const* a, double v, double* b) {
    __m128d va = _mm_loadu_pd(a);
    __m128d rv = _mm_set1_pd(v);

    __m128d added = _mm_add_pd(va, rv);

    _mm_storeu_pd(b, added);
}

void ai3_add_v(double const* a, double v, double* b) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d rv = _mm256_set1_pd(v);

    __m256d added = _mm256_add_pd(va, rv);

    _mm256_maskstore_pd(b, mask, added);
}

void ai4_add_v(double const* a, double v, double* b) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d rv = _mm256_set1_pd(v);

    __m256d added = _mm256_add_pd(va, rv);

    _mm256_storeu_pd(b, added);
}

void ai_add_v(double const* a, unsigned int n, double v, double* b) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                b[i] = a[i] + v;
                break;
            case 2:
                ai2_add_v(&a[i], v, &b[i]);
                break;
            case 3:
                ai3_add_v(&a[i], v, &b[i]);
                break;
            default:
                ai4_add_v(&a[i], v, &b[i]);
                break;
        }
    }
}

void ai2_mul_v(double const* a, double v, double* b) {
    __m128d ra = _mm_loadu_pd(a);
    __m128d rv = _mm_set1_pd(v);
    ra = _mm_mul_pd(ra, rv);
    _mm_storeu_pd(b, ra);
}

void ai3_mul_v(double const* a, double v, double* b) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
    __m256d ra = _mm256_maskload_pd(a, mask);
    __m256d rv = _mm256_set1_pd(v);
    ra = _mm256_mul_pd(ra, rv);
    _mm256_maskstore_pd(b, mask, ra);
}

void ai4_mul_v(double const* a, double v, double* b) {
    __m256d ra = _mm256_loadu_pd(a);
    __m256d rv = _mm256_set1_pd(v);
    ra = _mm256_mul_pd(ra, rv);
    _mm256_storeu_pd(b, ra);
}

void ai_mul_v(double const* a, unsigned int n, double v, double* b) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
    __m256d rv4 = _mm256_set1_pd(v);
    __m256d ra4;
    __m128d rv2;
    __m128d ra2;
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                b[i] = a[i] * v;
                break;
            case 2:
                ra2 = _mm_loadu_pd(&a[i]);
                rv2 = _mm_set1_pd(v);
                ra2 = _mm_mul_pd(ra2, rv2);
                _mm_storeu_pd(&b[i], ra2);
                break;
            case 3:
                ra4 = _mm256_maskload_pd(&a[i], mask);
                ra4 = _mm256_mul_pd(ra4, rv4);
                _mm256_maskstore_pd(&b[i], mask, ra4);
                break;
            default:
                ra4 = _mm256_loadu_pd(&a[i]);
                ra4 = _mm256_mul_pd(ra4, rv4);
                _mm256_storeu_pd(&b[i], ra4);
                break;
        }
    }
}

void Aij3x3_add_v(double const* A, double v, double* B) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d rv = _mm256_set1_pd(v);

    __m256d added0 = _mm256_add_pd(va0, rv);
    __m256d added1 = _mm256_add_pd(va1, rv);

    _mm256_storeu_pd(B, added0);
    _mm256_storeu_pd(&B[4], added1);
    B[8] = A[8] + v;
}

void Aij3x3_mul_v(double const* A, double v, double* B) {
    __m256d r0 = _mm256_loadu_pd(A);
    __m256d r1 = _mm256_loadu_pd(&A[4]);
    __m256d rv = _mm256_set1_pd(v);
    r0 = _mm256_mul_pd(r0, rv);
    r1 = _mm256_mul_pd(r1, rv);
    _mm256_storeu_pd(B, r0);
    _mm256_storeu_pd(&B[4], r1);
    B[8] = A[8] * v;
}

void ai2_add_bi(double const* a, double const* b, double* c) {
    __m128d va0 = _mm_loadu_pd(a);
    __m128d vb0 = _mm_loadu_pd(b);

    __m128d add0 = _mm_add_pd(va0, vb0);

    _mm_storeu_pd(c, add0);
}

void ai3_add_bi(double const* a, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d add = _mm256_add_pd(va, vb);

    _mm256_maskstore_pd(c, mask, add);
}

void ai4_add_bi(double const* a, double const* b, double* c) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d add = _mm256_add_pd(va, vb);

    _mm256_storeu_pd(c, add);
}

void ai_add_bi(double const* a, double const* b, unsigned int n, double* c) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                c[i] = a[i] + b[i];
                break;
            case 2:
                ai2_add_bi(&a[i], &b[i], &c[i]);
                break;
            case 3:
                ai3_add_bi(&a[i], &b[i], &c[i]);
                break;
            default:
                ai4_add_bi(&a[i], &b[i], &c[i]);
                break;
        }
    }
}

void ai2_add_ebi(double const* a, double e, double const* b, double* c) {
    __m128d va = _mm_loadu_pd(a);
    __m128d vb = _mm_loadu_pd(b);

    __m128d rv = _mm_set1_pd(e);
    vb = _mm_mul_pd(vb, rv);
    __m128d add0 = _mm_add_pd(va, vb);

    _mm_storeu_pd(c, add0);
}

void ai3_add_ebi(double const* a, double e, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d rv = _mm256_set1_pd(e);
    vb = _mm256_mul_pd(vb, rv);
    __m256d add = _mm256_add_pd(va, vb);

    _mm256_maskstore_pd(c, mask, add);
}

void ai4_add_ebi(double const* a, double e, double const* b, double* c) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d rv = _mm256_set1_pd(e);
    vb = _mm256_mul_pd(vb, rv);
    __m256d add = _mm256_add_pd(va, vb);

    _mm256_storeu_pd(c, add);
}

void ai_add_ebi(double const* a, double e, double const* b, unsigned int n,
                double* c) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                c[i] = a[i] + e * b[i];
                break;
            case 2:
                ai2_add_ebi(&a[i], e, &b[i], &c[i]);
                break;
            case 3:
                ai3_add_ebi(&a[i], e, &b[i], &c[i]);
                break;
            default:
                ai4_add_ebi(&a[i], e, &b[i], &c[i]);
                break;
        }
    }
}
void ai2_sub_bi(double const* a, double const* b, double* c) {
    __m128d va0 = _mm_loadu_pd(a);
    __m128d vb0 = _mm_loadu_pd(b);

    __m128d sub = _mm_sub_pd(va0, vb0);

    _mm_storeu_pd(c, sub);
}

void ai3_sub_bi(double const* a, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d sub = _mm256_sub_pd(va, vb);

    _mm256_maskstore_pd(c, mask, sub);
}

void ai4_sub_bi(double const* a, double const* b, double* c) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d sub = _mm256_sub_pd(va, vb);

    _mm256_storeu_pd(c, sub);
}

void ai_sub_bi(double const* a, double const* b, unsigned int n, double* c) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                c[i] = a[i] - b[i];
                break;
            case 2:
                ai2_sub_bi(&a[i], &b[i], &c[i]);
                break;
            case 3:
                ai3_sub_bi(&a[i], &b[i], &c[i]);
                break;
            default:
                ai4_sub_bi(&a[i], &b[i], &c[i]);
                break;
        }
    }
}

void ai2_mul_bi(double const* a, double const* b, double* c) {
    __m128d va0 = _mm_loadu_pd(a);
    __m128d vb0 = _mm_loadu_pd(b);

    __m128d rv = _mm_mul_pd(va0, vb0);

    _mm_storeu_pd(c, rv);
}

void ai3_mul_bi(double const* a, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d rv = _mm256_mul_pd(va, vb);

    _mm256_maskstore_pd(c, mask, rv);
}

void ai4_mul_bi(double const* a, double const* b, double* c) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d rv = _mm256_mul_pd(va, vb);

    _mm256_storeu_pd(c, rv);
}

void ai_mul_bi(double const* a, double const* b, unsigned int n, double* c) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                c[i] = a[i] * b[i];
                break;
            case 2:
                ai2_mul_bi(&a[i], &b[i], &c[i]);
                break;
            case 3:
                ai3_mul_bi(&a[i], &b[i], &c[i]);
                break;
            default:
                ai4_mul_bi(&a[i], &b[i], &c[i]);
                break;
        }
    }
}

void ai2_div_bi(double const* a, double const* b, double* c) {
    __m128d va0 = _mm_loadu_pd(a);
    __m128d vb0 = _mm_loadu_pd(b);

    __m128d rv = _mm_div_pd(va0, vb0);

    _mm_storeu_pd(c, rv);
}

void ai3_div_bi(double const* a, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d rv = _mm256_div_pd(va, vb);

    _mm256_maskstore_pd(c, mask, rv);
}

void ai4_div_bi(double const* a, double const* b, double* c) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d rv = _mm256_div_pd(va, vb);

    _mm256_storeu_pd(c, rv);
}

void ai_div_bi(double const* a, double const* b, unsigned int n, double* c) {
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                c[i] = a[i] / b[i];
                break;
            case 2:
                ai2_div_bi(&a[i], &b[i], &c[i]);
                break;
            case 3:
                ai3_div_bi(&a[i], &b[i], &c[i]);
                break;
            default:
                ai4_div_bi(&a[i], &b[i], &c[i]);
                break;
        }
    }
}

void Aij3x3_add_Bij(double const* A, double const* B, double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d add0 = _mm256_add_pd(va0, vb0);
    __m256d add1 = _mm256_add_pd(va1, vb1);

    _mm256_storeu_pd(C, add0);
    _mm256_storeu_pd(&C[4], add1);
    C[8] = A[8] + B[8];
}

void Aij3x3_sub_Bij(double const* A, double const* B, double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d sub0 = _mm256_sub_pd(va0, vb0);
    __m256d sub1 = _mm256_sub_pd(va1, vb1);

    _mm256_storeu_pd(C, sub0);
    _mm256_storeu_pd(&C[4], sub1);
    C[8] = A[8] - B[8];
}

void Aij3x3_mul_Bij(double const* A, double const* B, double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d rv0 = _mm256_mul_pd(va0, vb0);
    __m256d rv1 = _mm256_mul_pd(va1, vb1);

    _mm256_storeu_pd(C, rv0);
    _mm256_storeu_pd(&C[4], rv1);
    C[8] = A[8] * B[8];
}

void Aij3x3_div_Bij(double const* A, double const* B, double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d rv0 = _mm256_div_pd(va0, vb0);
    __m256d rv1 = _mm256_div_pd(va1, vb1);

    _mm256_storeu_pd(C, rv0);
    _mm256_storeu_pd(&C[4], rv1);
    C[8] = A[8] / B[8];
}

void Aij3x3_add_fBij(double const* A, double f, double const* B, double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d rv = _mm256_set1_pd(f);
    vb0 = _mm256_mul_pd(vb0, rv);
    vb1 = _mm256_mul_pd(vb1, rv);

    __m256d add0 = _mm256_add_pd(va0, vb0);
    __m256d add1 = _mm256_add_pd(va1, vb1);

    _mm256_storeu_pd(C, add0);
    _mm256_storeu_pd(&C[4], add1);

    C[8] = A[8] + f * B[8];
}

void eAij3x3_add_fBij(double e, double const* A, double f, double const* B,
                      double* C) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);

    __m256d rv0 = _mm256_set1_pd(e);
    __m256d rv1 = _mm256_set1_pd(f);

    va0 = _mm256_mul_pd(va0, rv0);
    va1 = _mm256_mul_pd(va1, rv0);
    vb0 = _mm256_mul_pd(vb0, rv1);
    vb1 = _mm256_mul_pd(vb1, rv1);

    __m256d add0 = _mm256_add_pd(va0, vb0);
    __m256d add1 = _mm256_add_pd(va1, vb1);

    _mm256_storeu_pd(C, add0);
    _mm256_storeu_pd(&C[4], add1);

    C[8] = e * A[8] + f * B[8];
}

void eAij3x3_add_fBij_add_gCij(double e, double const* A, double f,
                               double const* B, double g, double* C,
                               double* D) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d va1 = _mm256_loadu_pd(&A[4]);
    __m256d vb0 = _mm256_loadu_pd(B);
    __m256d vb1 = _mm256_loadu_pd(&B[4]);
    __m256d vc0 = _mm256_loadu_pd(C);
    __m256d vc1 = _mm256_loadu_pd(&C[4]);

    __m256d rv0 = _mm256_set1_pd(e);
    __m256d rv1 = _mm256_set1_pd(f);
    __m256d rv2 = _mm256_set1_pd(g);

    va0 = _mm256_mul_pd(va0, rv0);
    va1 = _mm256_mul_pd(va1, rv0);
    vb0 = _mm256_mul_pd(vb0, rv1);
    vb1 = _mm256_mul_pd(vb1, rv1);
    vc0 = _mm256_mul_pd(vc0, rv2);
    vc1 = _mm256_mul_pd(vc1, rv2);

    __m256d add0 = _mm256_add_pd(va0, vb0);
    __m256d add1 = _mm256_add_pd(va1, vb1);
    add0 = _mm256_add_pd(add0, vc0);
    add1 = _mm256_add_pd(add1, vc1);

    _mm256_storeu_pd(D, add0);
    _mm256_storeu_pd(&D[4], add1);

    D[8] = e * A[8] + f * B[8] + g * C[8];
}

void Aij3x3Bjk(double const* A, double const* B, double* C) {
    __m256d va00 = _mm256_broadcast_sd(&A[0]);
    __m256d va01 = _mm256_broadcast_sd(&A[1]);
    __m256d va02 = _mm256_broadcast_sd(&A[2]);
    __m256d va10 = _mm256_broadcast_sd(&A[3]);
    __m256d va11 = _mm256_broadcast_sd(&A[4]);
    __m256d va12 = _mm256_broadcast_sd(&A[5]);
    __m256d va20 = _mm256_broadcast_sd(&A[6]);
    __m256d va21 = _mm256_broadcast_sd(&A[7]);
    __m256d va22 = _mm256_broadcast_sd(&A[8]);

    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d vb0 = _mm256_maskload_pd(B, mask);
    __m256d vb1 = _mm256_maskload_pd(&B[3], mask);
    __m256d vb2 = _mm256_maskload_pd(&B[6], mask);

    // i=0
    __m256d ma = _mm256_mul_pd(va00, vb0);
    __m256d mb = _mm256_mul_pd(va01, vb1);
    __m256d mc = _mm256_mul_pd(va02, vb2);
    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(C, vc);
    // i=1
    ma = _mm256_mul_pd(va10, vb0);
    mb = _mm256_mul_pd(va11, vb1);
    mc = _mm256_mul_pd(va12, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(&C[3], vc);
    // i=2
    ma = _mm256_mul_pd(va20, vb0);
    mb = _mm256_mul_pd(va21, vb1);
    mc = _mm256_mul_pd(va22, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_maskstore_pd(&C[6], mask, vc);
}

void Aji3x3Bjk(double const* A, double const* B, double* C) {
    __m256d va00 = _mm256_broadcast_sd(&A[0]);
    __m256d va01 = _mm256_broadcast_sd(&A[3]);
    __m256d va02 = _mm256_broadcast_sd(&A[6]);
    __m256d va10 = _mm256_broadcast_sd(&A[1]);
    __m256d va11 = _mm256_broadcast_sd(&A[4]);
    __m256d va12 = _mm256_broadcast_sd(&A[7]);
    __m256d va20 = _mm256_broadcast_sd(&A[2]);
    __m256d va21 = _mm256_broadcast_sd(&A[5]);
    __m256d va22 = _mm256_broadcast_sd(&A[8]);

    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d vb0 = _mm256_maskload_pd(B, mask);
    __m256d vb1 = _mm256_maskload_pd(&B[3], mask);
    __m256d vb2 = _mm256_maskload_pd(&B[6], mask);

    // i=0
    __m256d ma = _mm256_mul_pd(va00, vb0);
    __m256d mb = _mm256_mul_pd(va01, vb1);
    __m256d mc = _mm256_mul_pd(va02, vb2);
    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(C, vc);
    // i=1
    ma = _mm256_mul_pd(va10, vb0);
    mb = _mm256_mul_pd(va11, vb1);
    mc = _mm256_mul_pd(va12, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(&C[3], vc);
    // i=2
    ma = _mm256_mul_pd(va20, vb0);
    mb = _mm256_mul_pd(va21, vb1);
    mc = _mm256_mul_pd(va22, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_maskstore_pd(&C[6], mask, vc);
}

void Aij3x3Bkj(double const* A, double const* B, double* C) {
    __m256d va00 = _mm256_broadcast_sd(&A[0]);
    __m256d va01 = _mm256_broadcast_sd(&A[1]);
    __m256d va02 = _mm256_broadcast_sd(&A[2]);
    __m256d va10 = _mm256_broadcast_sd(&A[3]);
    __m256d va11 = _mm256_broadcast_sd(&A[4]);
    __m256d va12 = _mm256_broadcast_sd(&A[5]);
    __m256d va20 = _mm256_broadcast_sd(&A[6]);
    __m256d va21 = _mm256_broadcast_sd(&A[7]);
    __m256d va22 = _mm256_broadcast_sd(&A[8]);

    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d vb0 = _mm256_set_pd(0, B[6], B[3], B[0]);
    __m256d vb1 = _mm256_set_pd(0, B[7], B[4], B[1]);
    __m256d vb2 = _mm256_set_pd(0, B[8], B[5], B[2]);

    // i=0
    __m256d ma = _mm256_mul_pd(va00, vb0);
    __m256d mb = _mm256_mul_pd(va01, vb1);
    __m256d mc = _mm256_mul_pd(va02, vb2);
    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(C, vc);
    // i=1
    ma = _mm256_mul_pd(va10, vb0);
    mb = _mm256_mul_pd(va11, vb1);
    mc = _mm256_mul_pd(va12, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(&C[3], vc);
    // i=2
    ma = _mm256_mul_pd(va20, vb0);
    mb = _mm256_mul_pd(va21, vb1);
    mc = _mm256_mul_pd(va22, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_maskstore_pd(&C[6], mask, vc);
}

void Aji3x3Bkj(double const* A, double const* B, double* C) {
    __m256d va00 = _mm256_broadcast_sd(&A[0]);
    __m256d va01 = _mm256_broadcast_sd(&A[3]);
    __m256d va02 = _mm256_broadcast_sd(&A[6]);
    __m256d va10 = _mm256_broadcast_sd(&A[1]);
    __m256d va11 = _mm256_broadcast_sd(&A[4]);
    __m256d va12 = _mm256_broadcast_sd(&A[7]);
    __m256d va20 = _mm256_broadcast_sd(&A[2]);
    __m256d va21 = _mm256_broadcast_sd(&A[5]);
    __m256d va22 = _mm256_broadcast_sd(&A[8]);

    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d vb0 = _mm256_set_pd(0, B[6], B[3], B[0]);
    __m256d vb1 = _mm256_set_pd(0, B[7], B[4], B[1]);
    __m256d vb2 = _mm256_set_pd(0, B[8], B[5], B[2]);

    // i=0
    __m256d ma = _mm256_mul_pd(va00, vb0);
    __m256d mb = _mm256_mul_pd(va01, vb1);
    __m256d mc = _mm256_mul_pd(va02, vb2);
    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(C, vc);
    // i=1
    ma = _mm256_mul_pd(va10, vb0);
    mb = _mm256_mul_pd(va11, vb1);
    mc = _mm256_mul_pd(va12, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_storeu_pd(&C[3], vc);
    // i=2
    ma = _mm256_mul_pd(va20, vb0);
    mb = _mm256_mul_pd(va21, vb1);
    mc = _mm256_mul_pd(va22, vb2);
    vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);
    _mm256_maskstore_pd(&C[6], mask, vc);
}

void Aij3x3bj(double const* A, double const* b, double* c) {
    __m256d va0 = _mm256_set_pd(0, A[6], A[3], A[0]);
    __m256d va1 = _mm256_set_pd(0, A[7], A[4], A[1]);
    __m256d va2 = _mm256_set_pd(0, A[8], A[5], A[2]);

    __m256d vb0 = _mm256_broadcast_sd(&b[0]);
    __m256d vb1 = _mm256_broadcast_sd(&b[1]);
    __m256d vb2 = _mm256_broadcast_sd(&b[2]);

    __m256d ma = _mm256_mul_pd(va0, vb0);
    __m256d mb = _mm256_mul_pd(va1, vb1);
    __m256d mc = _mm256_mul_pd(va2, vb2);

    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);

    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
    _mm256_maskstore_pd(c, mask, vc);
}

void Aji3x3bj(double const* A, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va0 = _mm256_maskload_pd(A, mask);
    __m256d va1 = _mm256_maskload_pd(&A[3], mask);
    __m256d va2 = _mm256_maskload_pd(&A[6], mask);

    __m256d vb0 = _mm256_broadcast_sd(&b[0]);
    __m256d vb1 = _mm256_broadcast_sd(&b[1]);
    __m256d vb2 = _mm256_broadcast_sd(&b[2]);

    __m256d ma = _mm256_mul_pd(va0, vb0);
    __m256d mb = _mm256_mul_pd(va1, vb1);
    __m256d mc = _mm256_mul_pd(va2, vb2);

    __m256d vc = _mm256_add_pd(ma, mb);
    vc = _mm256_add_pd(vc, mc);

    _mm256_maskstore_pd(c, mask, vc);
}

void ai3bj(double const* a, double const* b, double* C) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d vbj = _mm256_maskload_pd(b, mask);
    __m256d va0 = _mm256_broadcast_sd(&a[0]);
    __m256d va1 = _mm256_broadcast_sd(&a[1]);
    __m256d va2 = _mm256_broadcast_sd(&a[2]);

    __m256d m0 = _mm256_mul_pd(va0, vbj);
    __m256d m1 = _mm256_mul_pd(va1, vbj);
    __m256d m2 = _mm256_mul_pd(va2, vbj);

    _mm256_storeu_pd(C, m0);
    _mm256_storeu_pd(&C[3], m1);
    _mm256_maskstore_pd(&C[6], mask, m2);
}

void ai3bjcross(double const* a, double const* b, double* c) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va0 = _mm256_maskload_pd(a, mask);
    __m256d vb0 = _mm256_maskload_pd(b, mask);

    __m256d vap = _mm256_permute4x64_pd(va0, 0b11001001);
    __m256d van = _mm256_permute4x64_pd(va0, 0b11010010);
    __m256d vbp = _mm256_permute4x64_pd(vb0, 0b11010010);
    __m256d vbn = _mm256_permute4x64_pd(vb0, 0b11001001);

    __m256d mp = _mm256_mul_pd(vap, vbp);
    __m256d mn = _mm256_mul_pd(van, vbn);

    __m256d vc = _mm256_sub_pd(mp, mn);

    _mm256_maskstore_pd(c, mask, vc);
}

double ai2bi(double const* a, double const* b) {
    __m128d va = _mm_loadu_pd(a);
    __m128d vb = _mm_loadu_pd(b);

    __m128d m = _mm_mul_pd(va, vb);
    __m128d swapped = _mm_shuffle_pd(m, m, 0b01);
    __m128d sum = _mm_add_pd(m, swapped);

    return _mm_cvtsd_f64(sum);
}

double ai3bi(double const* a, double const* b) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d vb = _mm256_maskload_pd(b, mask);

    __m256d m = _mm256_mul_pd(va, vb);

    __m128d mlow = _mm256_castpd256_pd128(m);
    __m128d mhigh = _mm256_extractf128_pd(m, 1);
    __m128d sum = _mm_add_pd(mlow, mhigh);

    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);
    __m128d dotproduct = _mm_add_pd(sum, swapped);
    return _mm_cvtsd_f64(dotproduct);
}

double ai4bi(double const* a, double const* b) {
    __m256d va = _mm256_loadu_pd(a);
    __m256d vb = _mm256_loadu_pd(b);

    __m256d m = _mm256_mul_pd(va, vb);

    __m128d mlow = _mm256_castpd256_pd128(m);
    __m128d mhigh = _mm256_extractf128_pd(m, 1);
    __m128d sum = _mm_add_pd(mlow, mhigh);

    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);
    __m128d dotproduct = _mm_add_pd(sum, swapped);
    return _mm_cvtsd_f64(dotproduct);
}

double aibi(double const* a, double const* b, unsigned int n) {
    double r{0};
    for (size_t i{0}; i < n; i += 4) {
        size_t diff{n - i};
        switch (diff) {
            case 1:
                r += a[i] * b[i];
                break;
            case 2:
                r += ai2bi(&a[i], &b[i]);
                break;
            case 3:
                r += ai3bi(&a[i], &b[i]);
                break;
            default:
                r += ai4bi(&a[i], &b[i]);
                break;
        }
    }
    return r;
}

double ai3norm(double const* a) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d ma = _mm256_mul_pd(va, va);

    __m128d mlow = _mm256_castpd256_pd128(ma);
    __m128d mhigh = _mm256_extractf128_pd(ma, 1);
    __m128d sum = _mm_add_pd(mlow, mhigh);
    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);

    __m128d dotproduct = _mm_add_pd(sum, swapped);
    return sqrt(_mm_cvtsd_f64(dotproduct));
}

void ai3unit(double const* a, double* b) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
    __m256d va = _mm256_maskload_pd(a, mask);
    __m256d ma = _mm256_mul_pd(va, va);

    __m128d mlow = _mm256_castpd256_pd128(ma);
    __m128d mhigh = _mm256_extractf128_pd(ma, 1);
    __m128d sum = _mm_add_pd(mlow, mhigh);
    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);
    __m128d dotproduct = _mm_add_pd(sum, swapped);

    double inv_norm = 1.0 / sqrt(_mm_cvtsd_f64(dotproduct));
    __m256d rin = _mm256_set1_pd(inv_norm);

    __m256d vb = _mm256_mul_pd(va, rin);
    _mm256_maskstore_pd(b, mask, vb);
}

double Aij3x3norm(double const* A) {
    __m256d va0 = _mm256_loadu_pd(A);
    __m256d va1 = _mm256_loadu_pd(&A[4]);

    __m256d m0 = _mm256_mul_pd(va0, va0);
    __m256d m1 = _mm256_mul_pd(va1, va1);

    __m128d mlow = _mm256_castpd256_pd128(m0);
    __m128d mhigh = _mm256_extractf128_pd(m0, 1);
    __m128d sum = _mm_add_pd(mlow, mhigh);
    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);

    __m128d dotproduct = _mm_add_pd(sum, swapped);

    mlow = _mm256_castpd256_pd128(m1);
    mhigh = _mm256_extractf128_pd(m1, 1);
    sum = _mm_add_pd(mlow, mhigh);
    swapped = _mm_shuffle_pd(sum, sum, 0b01);

    dotproduct = _mm_add_pd(dotproduct, sum);
    dotproduct = _mm_add_pd(dotproduct, swapped);
    return sqrt(_mm_cvtsd_f64(dotproduct) + A[8] * A[8]);
}

double Aij3x3det(double const* A) {
    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d v0 = _mm256_maskload_pd(A, mask);
    __m256d v1p = _mm256_maskload_pd(&A[3], mask);
    v1p = _mm256_permute4x64_pd(v1p, 0b11001001);
    __m256d v2p = _mm256_maskload_pd(&A[6], mask);
    v2p = _mm256_permute4x64_pd(v2p, 0b11010010);

    __m256d v1n = _mm256_maskload_pd(&A[3], mask);
    v1n = _mm256_permute4x64_pd(v1n, 0b11010010);
    __m256d v2n = _mm256_maskload_pd(&A[6], mask);
    v2n = _mm256_permute4x64_pd(v2n, 0b11001001);

    __m256d mp = _mm256_mul_pd(v0, v1p);
    mp = _mm256_mul_pd(mp, v2p);

    __m256d mn = _mm256_mul_pd(v0, v1n);
    mn = _mm256_mul_pd(mn, v2n);

    __m256d v = _mm256_sub_pd(mp, mn);

    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(vlow, vhigh);

    __m128d swapped = _mm_shuffle_pd(sum, sum, 0b01);
    __m128d det = _mm_add_pd(sum, swapped);
    return _mm_cvtsd_f64(det);
}

void Aij3x3trans(double* A) {
    double tmp;
    tmp = A[1];
    A[1] = A[3];
    A[3] = tmp;
    tmp = A[2];
    A[2] = A[6];
    A[6] = tmp;
    tmp = A[5];
    A[5] = A[7];
    A[7] = tmp;
}

void Aij3x3trans(double const* A, double* B) {
    B[0] = A[0];
    B[1] = A[3];
    B[2] = A[6];
    B[3] = A[1];
    B[4] = A[4];
    B[5] = A[7];
    B[6] = A[2];
    B[7] = A[5];
    B[8] = A[8];
}

double Aii2x2(double const* A) { return A[0] + A[3]; }
double Aii3x3(double const* A) { return A[0] + A[4] + A[8]; }

void Aij3x3inv(double const* A, double* B) {
    // c=0:3
    __m256d r0 = _mm256_set_pd(A[4], A[1], A[7], A[4]);
    __m256d r1 = _mm256_set_pd(A[2], A[5], A[5], A[8]);
    __m256d m01 = _mm256_mul_pd(r0, r1);
    __m256d sm01 = _mm256_permute_pd(m01, 0b0101);
    __m256d ps01 = _mm256_sub_pd(m01, sm01);

    __m256d r2 = _mm256_set_pd(A[6], A[3], A[7], A[1]);
    __m256d r3 = _mm256_set_pd(A[5], A[8], A[2], A[8]);
    __m256d m23 = _mm256_mul_pd(r2, r3);
    __m256d sm23 = _mm256_permute_pd(m23, 0b0101);
    __m256d ns23 = _mm256_sub_pd(m23, sm23);

    __m256d c0 = _mm256_shuffle_pd(ps01, ns23, 0b1010);

    // c=4:7
    r0 = _mm256_set_pd(A[6], A[3], A[6], A[0]);
    r1 = _mm256_set_pd(A[4], A[7], A[2], A[8]);
    m01 = _mm256_mul_pd(r0, r1);
    sm01 = _mm256_permute_pd(m01, 0b0101);
    ps01 = _mm256_sub_pd(m01, sm01);

    r2 = _mm256_set_pd(A[6], A[0], A[3], A[0]);
    r3 = _mm256_set_pd(A[1], A[7], A[2], A[5]);
    m23 = _mm256_mul_pd(r2, r3);
    sm23 = _mm256_permute_pd(m23, 0b0101);
    ns23 = _mm256_sub_pd(m23, sm23);

    __m256d c1 = _mm256_shuffle_pd(ps01, ns23, 0b1010);

    // c=8
    __m128d r4 = _mm_set_pd(A[3], A[0]);
    __m128d r5 = _mm_set_pd(A[1], A[4]);
    __m128d m45 = _mm_mul_pd(r4, r5);
    __m128d sm45 = _mm_permute_pd(m45, 0b01);
    __m128d ps45 = _mm_sub_pd(m45, sm45);

    double c8 = _mm_cvtsd_f64(ps45);

    double const inv_det = 1.0 / Aij3x3det(A);
    __m256d rid = _mm256_set1_pd(inv_det);

    c0 = _mm256_mul_pd(c0, rid);
    c1 = _mm256_mul_pd(c1, rid);
    _mm256_storeu_pd(B, c0);
    _mm256_storeu_pd(&B[4], c1);
    B[8] = c8 * inv_det;
}

void Rji3x3AjkRkl(double const* R, double const* A, double* B) {
    Aji3x3Bjk(R, A, B);
    Aij3x3Bjk(B, R, B);
}

// Based on the following work:
// Eberly, D. (2014) A Robust Eigensolver for 3 × 3 Symmetric Matrices.
// Available at:
// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf.
void Aij3x3eigen_sym(double const* A, double* e_val, double* E_vec) {
    auto idx = [](size_t i, size_t j) { return i * 3 + j; };
    auto calc_cos_sin = [](double const u, double const v, double& c,
                           double& s) {
        double lengh = sqrt(u * u + v * v);
        if (lengh <= 0.0) {
            c = -1.0;
            s = 0.0;
            return;
        }
        c = -abs(u) / lengh;
        s = -v / lengh;
        if (std::signbit(u)) s = -s;
    };
    auto test_convergence = [](double const b_1, double const b_2,
                               double const b_superdiag) {
        double soma{abs(b_1) + abs(b_2)};
        return soma + abs(b_superdiag) == soma;
    };

    double c;
    double s;
    calc_cos_sin(A[idx(1, 2)], -A[idx(0, 2)], c, s);
    double H[9] = {c, s, 0, s, -c, 0, 0, 0, 1};
    double B[9];
    Rji3x3AjkRkl(H, A, B);

    if (abs(B[idx(1, 2)]) <= abs(B[idx(0, 1)])) {
        double c2;
        double s2;
        int32_t power;
        frexp(B[idx(1, 2)], &power);
        uint32_t alpha{1074};
        uint32_t max_iter{2 * (power + alpha + 1)};

        for (uint32_t i = 0; i < max_iter; i++) {
            calc_cos_sin((B[idx(1, 1)] - B[idx(0, 0)]) / 2.0, -B[idx(0, 1)], c2,
                         s2);
            s = sqrt((1.0 - c2) / 2.0);
            c = s2 / (2.0 * s);

            double G[9] = {c, 0, -s, s, 0, c, 0, 1, 0};
            Rji3x3AjkRkl(G, B, B);
            Aij3x3Bjk(H, G, H);

            if (test_convergence(B[idx(0, 0)], B[idx(1, 1)], B[idx(0, 1)])) {
                calc_cos_sin((B[idx(1, 1)] - B[idx(0, 0)]) / 2.0, -B[idx(0, 1)],
                             c2, s2);
                s = sqrt((1.0 - c2) / 2.0);
                c = s2 / (2.0 * s);
                double H1[9] = {c, s, 0, s, -c, 0, 0, 0, 1};
                Rji3x3AjkRkl(H1, B, B);
                Aij3x3Bjk(H, H1, H);
                break;
            }
        }
    } else {
        double c2;
        double s2;
        int32_t power;
        frexp(B[idx(0, 1)], &power);
        uint32_t alpha{1074};
        uint32_t max_iter{2 * (power + alpha + 1)};

        for (uint32_t i = 0; i < max_iter; i++) {
            calc_cos_sin((B[idx(2, 2)] - B[idx(1, 1)]) / 2.0, -B[idx(1, 2)], c2,
                         s2);
            s = sqrt((1.0 - c2) / 2.0);
            c = s2 / (2.0 * s);

            double G[9] = {0, 1, 0, c, 0, -s, s, 0, c};
            Rji3x3AjkRkl(G, B, B);
            Aij3x3Bjk(H, G, H);

            if (test_convergence(B[idx(1, 1)], B[idx(2, 2)], B[idx(1, 2)])) {
                calc_cos_sin((B[idx(2, 2)] - B[idx(1, 1)]) / 2.0, -B[idx(1, 2)],
                             c2, s2);
                s = sqrt((1.0 - c2) / 2.0);
                c = s2 / (2.0 * s);
                double H1[9] = {1, 0, 0, 0, c, s, 0, s, -c};
                Rji3x3AjkRkl(H1, B, B);
                Aij3x3Bjk(H, H1, H);
                break;
            }
        }
    }

    e_val[0] = B[0];
    e_val[1] = B[4];
    e_val[2] = B[8];
    cpyarr(H, 9, E_vec);
}

}  // namespace gel
