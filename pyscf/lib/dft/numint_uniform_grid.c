/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Fast numerical integration on uniform grids.
 * (See also cp2k multigrid algorithm)
 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/ft_ao.h"
#include "gto/grid_ao_drv.h"
#include "vhf/fblas.h"

#ifndef __USE_ISOC99
#define rint(x) (int)round(x)
#endif

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3
#define OF_CMPLX        2
#define EXPCUTOFF15     40
#define EXPMAX          700
#define EXPMIN          -700
#define MAX_THREADS     256

#define SQUARE(x)       (*(x) * *(x) + *(x+1) * *(x+1) + *(x+2) * *(x+2))

double CINTsquare_dist(const double *r1, const double *r2);
double CINTcommon_fac_sp(int l);
void c2s_sph_1e(double *opij, double *gctr, int *dims,
                CINTEnvVars *envs, double *cache);
void c2s_cart_1e(double *opij, double *gctr, int *dims,
                 CINTEnvVars *envs, double *cache);

int CINTinit_int1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);

static const int _LEN_CART[] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};
static const int _CUM_LEN_CART[] = {
        1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816,
};
static int _MAX_RR_SIZE[] = {
        1, 4, 12, 30, 60, 120, 210, 350, 560, 840, 1260, 1800, 2520, 3465, 4620,
        6160, 8008, 10296, 13104, 16380, 20475,
};
#define STARTX_IF_L_DEC1(l)     0
#define STARTY_IF_L_DEC1(l)     (((l)<2)?0:_LEN_CART[(l)-2])
#define STARTZ_IF_L_DEC1(l)     (_LEN_CART[(l)-1]-1)

void GTOplain_vrr2d_ket_inc1(double *out, const double *g,
                             double *rirj, int li, int lj);
/* (li+lj,0) => (li,lj) */
// Input g is used as buffer in the iterations.
// Ensure size of g > _MAX_RR_SIZE[li+lj]
static void _plain_vrr2d(double *out, double *g, double *gbuf2, int li, int lj,
                         double *ri, double *rj)
{
        const int nmax = li + lj;
        double *g00, *g01, *gswap, *pg00, *pg01;
        int row_01, col_01, row_00, col_00;
        int i, j;
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];

        g00 = gbuf2;
        g01 = g;
        for (j = 1; j < lj; j++) {
                gswap = g00;
                g00 = g01;
                g01 = gswap;
                pg00 = g00;
                pg01 = g01;
                for (i = li; i <= nmax-j; i++) {
                        GTOplain_vrr2d_ket_inc1(pg01, pg00, rirj, i, j);
                        row_01 = _LEN_CART[i];
                        col_01 = _LEN_CART[j];
                        row_00 = _LEN_CART[i  ];
                        col_00 = _LEN_CART[j-1];
                        pg00 += row_00*col_00;
                        pg01 += row_01*col_01;
                }
        }
        GTOplain_vrr2d_ket_inc1(out, g01, rirj, li, lj);
}

/*
 * rcut is the distance over which the integration (from rcut to infty) is
 * smaller than the required precision
 * integral ~= \int_{rcut}^infty r^{l+2} exp(-alpha r^2) dr
 *
 * * if l is odd:
 *   integral = \sum_n (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n
 *                     * exp(-alpha {rcut}^2)
 *
 * * elif l is even and rcut > 1:
 *   integral < [\sum_{n<=l/2+1} (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n
 *               + 1/(2 alpha)^(l/2+2)] * exp(-alpha {rcut}^2)
 *
 * * elif l is even and rcut < 1:
 *   integral < [\sum_{n<=l/2+1} (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n] * exp(-alpha {rcut}^2)
 *              + (l+1)!! / (2 alpha)^{l/2+1} * \sqrt(pi/alpha)/2
 */
static double gto_rcut(double alpha, int l, double c, double log_prec)
{
        // Add penalty 1e-2 for other integral factors and coefficients
        log_prec -= 5 + log(4*M_PI);

        double log_c = log(fabs(c));
        double prod = 0;
        double r = 5.;
        double log_2a = log(2*alpha);
        double log_r = log(r);

        if (2*log_r + log_2a > 1) { // r^2 >~ 3/(2a)
                prod = (l+1) * log_r - log_2a;
        } else {
                prod = -(l+4)/2 * log_2a;
        }

        //log_r = .5 * (prod / alpha);
        //if (2*log_r + log_2a > 1) {
        //        prod = (l+1) * log_r - log_2a;
        //} else {
        //        prod = -(l+4)/2 * log_2a;
        //}

        prod += log_c - log_prec;
        if (prod < alpha) {
                // if rcut < 1, estimating based on exp^{-a*rcut^2}
                prod = log_c - log_prec;
        }
        if (prod > 0) {
                r = sqrt(prod / alpha);
        } else {
                r = 0;
        }
        return r;
}

static int _has_overlap(int nx0, int nx1, int nx_per_cell)
{
        return nx0 + 3 < nx1;
}

static int _num_grids_on_x(int nimgx, int nx0, int nx1, int nx_per_cell)
{
        int ngridx;
        if (nimgx == 1) {
                ngridx = nx1 - nx0;
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, nx_per_cell)) {
                ngridx = nx1 - nx0 + nx_per_cell;
        } else {
                ngridx = nx_per_cell;
        }
        return ngridx;
}

static int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                            double a, double b, double cutoff,
                            double xi, double xj, double ai, double aj,
                            int periodic, int nx_per_cell, int topl, double *cache)
{
        double aij = ai + aj;
        double xij = (ai * xi + aj * xj) / aij;
        double heights_inv = b;
        double xij_frac = xij * heights_inv;
        double edge0 = xij_frac - cutoff * heights_inv;
        double edge1 = xij_frac + cutoff * heights_inv;

        int nimg0 = 0;
        int nimg1 = 1;
        if (periodic) {
                nimg0 = (int)floor(edge0);
                nimg1 = (int)ceil (edge1);
        }

        int nx0 = (int)floor(edge0 * nx_per_cell);
        int nx1 = (int)ceil (edge1 * nx_per_cell);
        // to ensure nx0, nx1 in unit cell
        if (periodic) {
                nx0 = (nx0 + nimg1 * nx_per_cell) % nx_per_cell;
                nx1 = (nx1 + nimg1 * nx_per_cell) % nx_per_cell;
        } else {
                nx0 = MIN(nx0, nx_per_cell);
                nx0 = MAX(nx0, 0);
                nx1 = MIN(nx1, nx_per_cell);
                nx1 = MAX(nx1, 0);
        }
        img_slice[0] = nimg0;
        img_slice[1] = nimg1;
        grid_slice[0] = nx0;
        grid_slice[1] = nx1;

        int nimg = nimg1 - nimg0;
        int nmx = nimg * nx_per_cell;
        int ngridx = _num_grids_on_x(nimg, nx0, nx1, nx_per_cell);
        if (ngridx == 0) {
                return 0;
        }

        int i, m, l;
        double *px0;

        double *gridx = cache;
        double *xs_all = cache + nimg * nx_per_cell;
        int grid_close_to_xij = rint(xij_frac * nx_per_cell);
        if (nimg == 1) {
                xs_all = xs_exp;
                grid_close_to_xij = MIN(grid_close_to_xij, nx_per_cell);
                grid_close_to_xij = MAX(grid_close_to_xij, 0);
        }

        double img0_x = a * nimg0;
        double dx = a / nx_per_cell;
        double base_x = img0_x + dx * grid_close_to_xij;
        double x0xij = base_x - xij;
        double _x0x0 = -aij * x0xij * x0xij;
        if (_x0x0 < EXPMIN) {
                return 0;
        }

        double _dxdx = -aij * dx * dx;
        double _x0dx = -2 * aij * x0xij * dx;
        double exp_dxdx = exp(_dxdx);
        double exp_2dxdx = exp_dxdx * exp_dxdx;
        double exp_x0dx = exp(_x0dx + _dxdx);
        double exp_x0x0 = exp(_x0x0);

        for (i = grid_close_to_xij; i < nmx; i++) {
                xs_all[i] = exp_x0x0;
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
        }

        exp_x0dx = exp(_dxdx - _x0dx);
        exp_x0x0 = exp(_x0x0);
        for (i = grid_close_to_xij-1; i >= 0; i--) {
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
                xs_all[i] = exp_x0x0;
        }

        if (topl > 0) {
                double x0xi = img0_x - xi;
                for (i = 0; i < nmx; i++) {
                        gridx[i] = x0xi + i * dx;
                }
                for (l = 1; l <= topl; l++) {
                        px0 = xs_all + (l-1) * nmx;
                        for (i = 0; i < nmx; i++) {
                                px0[nmx+i] = px0[i] * gridx[i];
                        }
                }
        }

        if (nimg > 1) {
                for (l = 0; l <= topl; l++) {
                        px0 = xs_all + l * nmx;
                        for (i = 0; i < nx_per_cell; i++) {
                                xs_exp[i] = px0[i];
                        }
                        for (m = 1; m < nimg; m++) {
                                px0 = xs_all + l * nmx + m*nx_per_cell;
                                for (i = 0; i < nx_per_cell; i++) {
                                        xs_exp[l*nx_per_cell+i] += px0[i];
                                }
                        }
                }
        }
        return ngridx;
}

static void _orth_ints(double *out, double *weights,
                       int floorl, int topl, double fac,
                       int *mesh, int *img_slice, int *grid_slice, 
                       double *xs_exp, double *ys_exp, double *zs_exp,
                       double *cache)
{
        int l1 = topl + 1;
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;
        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridx = _num_grids_on_x(nimgx, nx0, nx1, mesh[0]);
        int ngridy = _num_grids_on_x(nimgy, ny0, ny1, mesh[1]);
        //int ngridz = _num_grids_on_x(nimgz, nz0, nz1, mesh[2]);

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        int xcols = mesh[1] * mesh[2];
        int ycols = mesh[2];
        double *weightyz = cache;
        double *weightz = weightyz + l1*xcols;
        double *pz, *pweightz;
        double val;
        int lx, ly, lz;
        int l, i, n;

        //TODO: optimize the case in which nimgy << mesh[1] and nimgz << mesh[2]
        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
                ngridx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D1, weightyz, &xcols);
        } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
        }

        if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                        // call _orth_dot_z if ngridz << nimgz
                }
        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
                ngridy = mesh[1] - ny0;
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ny1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D1, weightz+lx*l1*ycols, &ycols);
                        // call _orth_dot_z if ngridz << nimgz
                }
        } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        }

        if (nimgz == 1) {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = nz0; i < nz1; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        //TODO:} elif (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
        } else {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = 0; i < mesh[2]; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        }
}

void NUMINTeval_lda_orth(double *mat, int *dims, double fac, double log_prec,
                         int dimension, double *a, double *b, int *mesh,
                         double *weights, int *shls, int *atm, int *bas,
                         double *env, double *cache)
{
        int i_sh = shls[0];
        int j_sh = shls[1];
        int li = bas(ANG_OF, i_sh);
        int lj = bas(ANG_OF, j_sh);
        double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        double ai = env[bas(PTR_EXP, i_sh)];
        double aj = env[bas(PTR_EXP, j_sh)];
        double ci = env[bas(PTR_COEFF, i_sh)];
        double cj = env[bas(PTR_COEFF, j_sh)];
        double aij = ai + aj;
        double rrij = CINTsquare_dist(ri, rj);
        double eij = (ai * aj / aij) * rrij;
        fac *= exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < 1e-40) { //if (eij > EXPCUTOFF15) {
                return;
        }

        int floorl = li;
        int topl = li + lj;
        int l1 = topl + 1;
        int offset_g1d = _CUM_LEN_CART[floorl] - _LEN_CART[floorl];
        int len_g3d = _CUM_LEN_CART[topl] - offset_g1d;
        double *g3d = cache;
        double *xs_exp = g3d + len_g3d;
        double *ys_exp = xs_exp + l1 * mesh[0];
        double *zs_exp = ys_exp + l1 * mesh[1];
        cache = zs_exp + l1 * mesh[2];

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        int img_slice[6];
        int grid_slice[6];
        int ngridx = _orth_components(xs_exp, img_slice, grid_slice,
                                      a[0], b[0], cutoff, ri[0], rj[0], ai, aj,
                                      (dimension>=1), mesh[0], topl, cache);
        if (ngridx == 0) {
                return;
        }

        int ngridy = _orth_components(ys_exp, img_slice+2, grid_slice+2,
                                      a[4], b[4], cutoff, ri[1], rj[1], ai, aj,
                                      (dimension>=2), mesh[1], topl, cache);
        if (ngridy == 0) {
                return;
        }

        int ngridz = _orth_components(zs_exp, img_slice+4, grid_slice+4,
                                      a[8], b[8], cutoff, ri[2], rj[2], ai, aj,
                                      (dimension>=3), mesh[2], topl, cache);
        if (ngridz == 0) {
                return;
        }

        _orth_ints(g3d, weights, floorl, topl, fac, mesh, img_slice, grid_slice,
                   xs_exp, ys_exp, zs_exp, cache);

        cache = g3d + _MAX_RR_SIZE[topl];
        double *out = cache + _MAX_RR_SIZE[topl];
        _plain_vrr2d(out, g3d, cache, li, lj, ri, rj);

        int di = _LEN_CART[li];
        int dj = _LEN_CART[lj];
        int naoi = dims[0];
        int i, j;
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                mat[j*naoi+i] += out[j*di+i];
        } }
}


void GTOrr_nablax_i(double *out, double *li_up, double *li_down,
                    int li, int lj, double ai);
void GTOrr_nablay_i(double *out, double *li_up, double *li_down,
                    int li, int lj, double ai);
void GTOrr_nablaz_i(double *out, double *li_up, double *li_down,
                    int li, int lj, double ai);

static void _plain_vrr2d_updown(double *out_up, double *out_down,
                                double *g, double *gbuf2, int li, int lj,
                                double *ri, double *rj)
{
        int nmax = li + 1 + lj;
        int li_1 = MAX(li - 1, 0);
        double *g00, *g01, *gswap, *pg00, *pg01;
        int row_01, col_01, row_00, col_00;
        int i, j;
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];

        g00 = gbuf2;
        g01 = g;
        for (j = 1; j < lj; j++) {
                gswap = g00;
                g00 = g01;
                g01 = gswap;
                pg00 = g00;
                pg01 = g01;
                for (i = li_1; i <= nmax-j; i++) {
                        GTOplain_vrr2d_ket_inc1(pg01, pg00, rirj, i, j);
                        row_01 = _LEN_CART[i];
                        col_01 = _LEN_CART[j];
                        row_00 = _LEN_CART[i  ];
                        col_00 = _LEN_CART[j-1];
                        pg00 += row_00*col_00;
                        pg01 += row_01*col_01;
                }
        }

        if (li == 0) {
                g01 += _LEN_CART[MAX(lj-1, 0)];
        } else {
                GTOplain_vrr2d_ket_inc1(out_down, g01, rirj, li_1, lj);
                g01 += (_LEN_CART[li_1] + _LEN_CART[li]) * _LEN_CART[MAX(lj-1, 0)];
        }
        GTOplain_vrr2d_ket_inc1(out_up, g01, rirj, li+1, lj);
}

void NUMINTeval_gga_orth(double *mat, int *dims, double fac, double log_prec,
                         int dimension, double *a, double *b, int *mesh,
                         double *weights, int *shls, int *atm, int *bas,
                         double *env, double *cache)
{
        int i_sh = shls[0];
        int j_sh = shls[1];
        int li = bas(ANG_OF, i_sh);
        int lj = bas(ANG_OF, j_sh);
        double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        double ai = env[bas(PTR_EXP, i_sh)];
        double aj = env[bas(PTR_EXP, j_sh)];
        double ci = env[bas(PTR_COEFF, i_sh)];
        double cj = env[bas(PTR_COEFF, j_sh)];
        double aij = ai + aj;
        double rrij = CINTsquare_dist(ri, rj);
        double eij = (ai * aj / aij) * rrij;
        fac *= exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < 1e-40) { //if (eij > EXPCUTOFF15) {
                return;
        }

        int floorl = MAX(li - 1, 0);
        int topl = li + 1 + lj;
        int l1 = topl + 1;
        int di = _LEN_CART[li];
        int dj = _LEN_CART[lj];
        double *out = cache;
        double *out_up = out + di * dj;
        double *out_down = out_up + _LEN_CART[li+1] * dj;
        double *g3d = out_down + di * dj;
        double *xs_exp = g3d + _MAX_RR_SIZE[topl];
        double *ys_exp = xs_exp + l1 * mesh[0];
        double *zs_exp = ys_exp + l1 * mesh[1];
        cache = zs_exp + l1 * mesh[2];

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        int img_slice[6];
        int grid_slice[6];
        int ngridx = _orth_components(xs_exp, img_slice, grid_slice,
                                      a[0], b[0], cutoff, ri[0], rj[0], ai, aj,
                                      (dimension>=1), mesh[0], topl, cache);
        if (ngridx == 0) {
                return;
        }

        int ngridy = _orth_components(ys_exp, img_slice+2, grid_slice+2,
                                      a[4], b[4], cutoff, ri[1], rj[1], ai, aj,
                                      (dimension>=2), mesh[1], topl, cache);
        if (ngridy == 0) {
                return;
        }

        int ngridz = _orth_components(zs_exp, img_slice+4, grid_slice+4,
                                      a[8], b[8], cutoff, ri[2], rj[2], ai, aj,
                                      (dimension>=3), mesh[2], topl, cache);
        if (ngridz == 0) {
                return;
        }

        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];
        double *vx = weights + ngrids;
        double *vy = vx + ngrids;
        double *vz = vy + ngrids;
        _orth_ints(g3d, weights, li, topl, fac, mesh, img_slice, grid_slice,
                   xs_exp, ys_exp, zs_exp, cache);
        _plain_vrr2d(out, g3d, cache, li, lj, ri, rj);

        _orth_ints(g3d, vx, floorl, topl, fac, mesh, img_slice, grid_slice,
                   xs_exp, ys_exp, zs_exp, cache);
        _plain_vrr2d_updown(out_up, out_down, g3d, cache, li, lj, ri, rj);
        GTOrr_nablax_i(out, out_up, out_down, li, lj, ai);

        _orth_ints(g3d, vy, floorl, topl, fac, mesh, img_slice, grid_slice,
                   xs_exp, ys_exp, zs_exp, cache);
        _plain_vrr2d_updown(out_up, out_down, g3d, cache, li, lj, ri, rj);
        GTOrr_nablay_i(out, out_up, out_down, li, lj, ai);

        _orth_ints(g3d, vz, floorl, topl, fac, mesh, img_slice, grid_slice,
                   xs_exp, ys_exp, zs_exp, cache);
        _plain_vrr2d_updown(out_up, out_down, g3d, cache, li, lj, ri, rj);
        GTOrr_nablaz_i(out, out_up, out_down, li, lj, ai);

        int naoi = dims[0];
        int i, j;
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                mat[j*naoi+i] += out[j*di+i];
        } }
}

static int _orth_cache_size(int *mesh, int l)
{
        int dcart = _LEN_CART[l];
        int deriv = 1;
        int topl = l + l + deriv;
        int l1 = topl + 1;
        int cache_size = 0;
        cache_size += l1 *(mesh[0] + mesh[1] + mesh[2]);
        cache_size += l1 * mesh[1] * mesh[2];
        cache_size += l1 * l1 * mesh[2];
        cache_size = MAX(cache_size, _MAX_RR_SIZE[topl] * 2);
        cache_size += _CUM_LEN_CART[topl] * 4;
        return dcart*dcart + cache_size;
}



static int _MAX_AFFINE_SIZE[] = {
        1, 8, 32, 108, 270, 640, 1280, 2500, 4375, 7560, 12096, 19208, 28812,
        43008, 61440, 87480,
};
/*
 * x = a00 x' + a10 y' + a20 z'
 * y = a01 x' + a11 y' + a21 z'
 * z = a02 x' + a12 y' + a22 z'
 * Given f(x',y',z') use the above equations to evaluate f(x,y,z)
 */
static void _affine_trans(double *out, double *int3d, double *a,
                          int floorl, int topl, double *cache)
{
        if (topl == 0) {
                out[0] = int3d[0];
                return;
        }

        int lx, ly, lz, l, m, n, i;
        int l1, l1l1, l1l1l1, lll;
        double *old = int3d;
        double *new = cache + _MAX_AFFINE_SIZE[topl];
        double *oldx, *oldy, *oldz, *newx, *tmp;
        double vx, vy, vz;

        if (floorl == 0) {
                out[0] = int3d[0];
                out += 1;
        }

        for (m = 1, l = topl; m <= topl; m++, l--) {
                l1 = l + 1;
                l1l1 = l1 * l1;
                lll = l * l * l;
                l1l1l1 = l1l1 * l1;
                newx = new;
                // attach x
                for (i = STARTX_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        oldx = old + i * l1l1l1 + l1l1;
                        oldy = old + i * l1l1l1 + l1;
                        oldz = old + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                vx = oldx[lx*l1l1+ly*l1+lz];
                                vy = oldy[lx*l1l1+ly*l1+lz];
                                vz = oldz[lx*l1l1+ly*l1+lz];
                                newx[n] = vx * a[0] + vy * a[3] + vz * a[6];
                        } } }
                        newx += lll;
                }

                // attach y
                for (i = STARTY_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        oldx = old + i * l1l1l1 + l1l1;
                        oldy = old + i * l1l1l1 + l1;
                        oldz = old + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                vx = oldx[lx*l1l1+ly*l1+lz];
                                vy = oldy[lx*l1l1+ly*l1+lz];
                                vz = oldz[lx*l1l1+ly*l1+lz];
                                newx[n] = vx * a[1] + vy * a[4] + vz * a[7];
                        } } }
                        newx += lll;
                }

                // attach z
                i = STARTZ_IF_L_DEC1(m);
                oldx = old + i * l1l1l1 + l1l1;
                oldy = old + i * l1l1l1 + l1;
                oldz = old + i * l1l1l1 + 1;
                for (n = 0, lx = 0; lx < l; lx++) {
                for (ly = 0; ly < l; ly++) {
                for (lz = 0; lz < l; lz++, n++) {
                        vx = oldx[lx*l1l1+ly*l1+lz];
                        vy = oldy[lx*l1l1+ly*l1+lz];
                        vz = oldz[lx*l1l1+ly*l1+lz];
                        newx[n] = vx * a[2] + vy * a[5] + vz * a[8];
                } } }

                if (floorl <= m) {
                        for (i = 0; i < _LEN_CART[m]; i++) {
                                out[i] = new[i * lll];
                        }
                        out += _LEN_CART[m];
                }

                if (m == 1) {
                        old = new;
                        new = cache;
                } else {
                        tmp = old;
                        old = new;
                        new = tmp;
                }
        }
}

static void _reverse_affine_trans(double *out3d, double *in, double *a,
                                  int floorl, int topl, double *cache)
{
        if (topl == 0) {
                out3d[0] = in[0];
                return;
        }

        int lx, ly, lz, l, m, n, i;
        int l1, l1l1, l1l1l1, lll;
        double *cart = in;
        double *old = cache;
        double *new = cache + _MAX_AFFINE_SIZE[topl];
        double *oldx, *newx, *newy, *newz, *tmp;

        for (l = floorl; l <= topl; l++) {
                cart += _LEN_CART[l];
        }
        for (l = 1, m = topl; l <= topl; l++, m--) {
                l1 = l + 1;
                l1l1 = l1 * l1;
                lll = l * l * l;
                l1l1l1 = l1l1 * l1;

                if (l == topl) {
                        new = out3d;
                }
                for (n = 0; n < l1l1l1*_LEN_CART[m-1]; n++) {
                        new[n] = 0;
                }

                if (floorl <= m) {
                        cart -= _LEN_CART[m];
                        for (i = 0; i < _LEN_CART[m]; i++) {
                                old[i * lll] = cart[i];
                        }
                }

                oldx = old;
                // attach x
                for (i = STARTX_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        newx = new + i * l1l1l1 + l1l1;
                        newy = new + i * l1l1l1 + l1;
                        newz = new + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                newx[lx*l1l1+ly*l1+lz] += a[0] * oldx[n];
                                newy[lx*l1l1+ly*l1+lz] += a[3] * oldx[n];
                                newz[lx*l1l1+ly*l1+lz] += a[6] * oldx[n];
                        } } }
                        oldx += lll;
                }

                // attach y
                for (i = STARTY_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        newx = new + i * l1l1l1 + l1l1;
                        newy = new + i * l1l1l1 + l1;
                        newz = new + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                newx[lx*l1l1+ly*l1+lz] += a[1] * oldx[n];
                                newy[lx*l1l1+ly*l1+lz] += a[4] * oldx[n];
                                newz[lx*l1l1+ly*l1+lz] += a[7] * oldx[n];
                        } } }
                        oldx += lll;
                }

                // attach z
                i = STARTZ_IF_L_DEC1(m);
                newx = new + i * l1l1l1 + l1l1;
                newy = new + i * l1l1l1 + l1;
                newz = new + i * l1l1l1 + 1;
                for (n = 0, lx = 0; lx < l; lx++) {
                for (ly = 0; ly < l; ly++) {
                for (lz = 0; lz < l; lz++, n++) {
                        newx[lx*l1l1+ly*l1+lz] += a[2] * oldx[n];
                        newy[lx*l1l1+ly*l1+lz] += a[5] * oldx[n];
                        newz[lx*l1l1+ly*l1+lz] += a[8] * oldx[n];
                } } }

                tmp = new;
                new = old;
                old = tmp;
        }

        if (floorl == 0) {
                out3d[0] = in[0];
        }
}

static void _nonorth_components(double *xs_exp, int *img_slice, int *grid_slice,
                                int periodic, int nx_per_cell, int topl,
                                int offset, int ngrid,
                                double xi_frac, double xij_frac, double cutoff,
                                double heights_inv)
{
        double edge0 = xij_frac - cutoff * heights_inv;
        double edge1 = xij_frac + cutoff * heights_inv;

        int nimg0 = 0;
        int nimg1 = 1;
        if (periodic) {
                nimg0 = (int)floor(edge0);
                nimg1 = (int)ceil (edge1);
        }

        int nx0 = (int)floor(edge0 * nx_per_cell);
        int nx1 = (int)ceil (edge1 * nx_per_cell);
        if (!periodic) {
                // to ensure nx0, nx1 in unit cell
                nx0 = MIN(nx0, nx_per_cell);
                nx0 = MAX(nx0, 0);
                nx1 = MIN(nx1, nx_per_cell);
                nx1 = MAX(nx1, 0);
        }
        if (nimg1 - nimg0 == 1) {
                nx0 = MAX(nx0, offset);
                nx1 = MIN(nx1, offset + ngrid);
        }

        img_slice[0] = nimg0;
        img_slice[1] = nimg1;
        grid_slice[0] = nx0;
        grid_slice[1] = nx1;

        int nx = nx1 - nx0;
        int i, l;
        double x0;
        double dx = 1. / nx_per_cell;
        double *pxs_exp;
        for (i = 0; i < nx; i++) {
                xs_exp[i] = 1;
        }
        for (l = 1; l <= topl; l++) {
                pxs_exp = xs_exp + (l-1) * nx;
                x0 = nx0 * dx - xi_frac;
                for (i = 0; i < nx; i++, x0+=dx) {
                        xs_exp[l*nx+i] = x0 * pxs_exp[i];
                }
        }
}

static void _nonorth_dot_z(double *val, double *weights,
                           int nz0, int nz1, int grid_close_to_zij,
                           double e_z0z0, double e_z0dz, double e_dzdz,
                           double _z0dz, double _dzdz)
{
        int iz;
        if (e_z0z0 == 0) {
                for (iz = 0; iz < nz1-nz0; iz++) {
                        val[iz] = 0;
                }
                return;
        }

        double exp_2dzdz = e_dzdz * e_dzdz;
        double exp_z0z0, exp_z0dz;

        exp_z0z0 = e_z0z0;
        exp_z0dz = e_z0dz * e_dzdz;
        for (iz = grid_close_to_zij; iz < nz1; iz++) {
                val[iz-nz0] = weights[iz] * exp_z0z0; //FIXME = weights[mod(iz,mesh[2])] * exp_z0z0;
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
        }

        exp_z0z0 = e_z0z0;
        if (e_z0dz != 0) {
                exp_z0dz = e_dzdz / e_z0dz;
        } else {
                exp_z0dz = exp(_dzdz - _z0dz);
        }
        for (iz = grid_close_to_zij-1; iz >= nz0; iz--) {
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
                val[iz-nz0] = weights[iz] * exp_z0z0; //FIXME = weights[mod(iz,mesh[2])] * exp_z0z0;
        }
}

void NUMINTeval_lda_nonorth(double *mat, int *dims, double fac, double log_prec,
                            int dimension, double *a, double *b, int *mesh,
                            double *weights, int *shls, int *atm, int *bas,
                            double *env, double *cache)
{
//FIXME: periodic condition
        int i_sh = shls[0];
        int j_sh = shls[1];
        int li = bas(ANG_OF, i_sh);
        int lj = bas(ANG_OF, j_sh);
        double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        double ai = env[bas(PTR_EXP, i_sh)];
        double aj = env[bas(PTR_EXP, j_sh)];
        double ci = env[bas(PTR_COEFF, i_sh)];
        double cj = env[bas(PTR_COEFF, j_sh)];
        double aij = ai + aj;
        double rrij = CINTsquare_dist(ri, rj);
        double eij = (ai * aj / aij) * rrij;
        fac *= exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < 1e-40) { //if (eij > EXPCUTOFF15) {
                return;
        }

        int floorl = li;
        int topl = li + lj;
        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int l1l1l1 = l1l1 * l1;
        int n;
        double *out = cache;
        cache += l1l1l1;

        double rij[3];
        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
        // rij_frac = einsum('ij,j->ik', b, rij)
        double xij_frac = rij[0] * b[0] + rij[1] * b[1] + rij[2] * b[2];
        double yij_frac = rij[0] * b[3] + rij[1] * b[4] + rij[2] * b[5];
        double zij_frac = rij[0] * b[6] + rij[1] * b[7] + rij[2] * b[8];
        double xi_frac = ri[0] * b[0] + ri[1] * b[1] + ri[2] * b[2];
        double yi_frac = ri[0] * b[3] + ri[1] * b[4] + ri[2] * b[5];
        double zi_frac = ri[0] * b[6] + ri[1] * b[7] + ri[2] * b[8];

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        double xheights_inv = sqrt(SQUARE(b  ));
        double yheights_inv = sqrt(SQUARE(b+3));
        double zheights_inv = sqrt(SQUARE(b+6));

        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;
        xs_exp = cache;
        _nonorth_components(xs_exp, img_slice, grid_slice, (dimension>=1),
                            mesh[0], topl, 0, mesh[0], xi_frac, xij_frac, cutoff,
                            xheights_inv);
        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ngridx = nx1 - nx0;
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgx = nimgx1 - nimgx0;
        if (nimgx == 1 && ngridx == 0) {
                return;
        }

        ys_exp = xs_exp + l1 * ngridx;
        _nonorth_components(ys_exp, img_slice+2, grid_slice+2, (dimension>=2),
                            mesh[1], topl, 0, mesh[1], yi_frac, yij_frac, cutoff,
                            yheights_inv);
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int ngridy = ny1 - ny0;
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgy = nimgy1 - nimgy0;
        if (nimgy == 1 && ngridy == 0) {
                return;
        }

        zs_exp = ys_exp + l1 * ngridy;
        _nonorth_components(zs_exp, img_slice+4, grid_slice+4, (dimension>=3),
                            mesh[2], topl, 0, mesh[2], zi_frac, zij_frac, cutoff,
                            zheights_inv);
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridz = nz1 - nz0;
        cache = zs_exp + l1 * ngridz;
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgz = nimgz1 - nimgz0;
        if (nimgz == 1 && ngridz == 0) {
                return;
        }

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        // aa = einsum('ij,kj->ik', a, a)
        //double aa[9];
        //int n3 = 3;
        //dgemm_(&TRANS_T, &TRANS_N, &n3, &n3, &n3,
        //       &aij, a, &n3, a, &n3, &D0, aa, &n3);
        double aa_xx = aij * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double aa_xy = aij * (a[0] * a[3] + a[1] * a[4] + a[2] * a[5]);
        double aa_xz = aij * (a[0] * a[6] + a[1] * a[7] + a[2] * a[8]);
        double aa_yy = aij * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]);
        double aa_yz = aij * (a[3] * a[6] + a[4] * a[7] + a[5] * a[8]);
        double aa_zz = aij * (a[6] * a[6] + a[7] * a[7] + a[8] * a[8]);

        int ix, iy;
        double dx = 1. / mesh[0];
        double dy = 1. / mesh[1];
        double dz = 1. / mesh[2];

        double *cache_xyz = cache;
        double *weight_x = cache_xyz + l1l1l1;
        double *weight_z = weight_x + l1l1 * ngridx;
        double *weight_yz = weight_z + l1 * ngridz;
        double *pweights;

        //int grid_close_to_xij = rint(xij_frac * mesh[0]);
        int grid_close_to_yij = rint(yij_frac * mesh[1]);
        int grid_close_to_zij = rint(zij_frac * mesh[2]);
        //if (dimension < 1) {
        //        grid_close_to_xij = MIN(grid_close_to_xij, mesh[0]);
        //        grid_close_to_xij = MAX(grid_close_to_xij, 0);
        //}
        if (dimension < 2) {
                grid_close_to_yij = MIN(grid_close_to_yij, mesh[1]);
                grid_close_to_yij = MAX(grid_close_to_yij, 0);
        }
        if (dimension < 3) {
                grid_close_to_zij = MIN(grid_close_to_zij, mesh[2]);
                grid_close_to_zij = MAX(grid_close_to_zij, 0);
        }

        double img0_x = 0;
        double img0_y = 0;
        double img0_z = 0;
        double base_x = img0_x;// + dx * grid_close_to_xij;
        double base_y = img0_y + dy * grid_close_to_yij;
        double base_z = img0_z + dz * grid_close_to_zij;
        double x0xij = base_x - xij_frac;
        double y0yij = base_y - yij_frac;
        double z0zij = base_z - zij_frac;

        double _dydy = -dy * dy * aa_yy;
        double _dzdz = -dz * dz * aa_zz;
        double _dydz = -dy * dz * aa_yz * 2;
        double exp_dydy = exp(_dydy);
        double exp_2dydy = exp_dydy * exp_dydy;
        double exp_dzdz = exp(_dzdz);
        double exp_dydz = exp(_dydz);
        double x1xij, tmpx, tmpy, tmpz;
        double _xyz0xyz0, _xyz0dy, _xyz0dz, _z0dz;
        double exp_xyz0xyz0, exp_xyz0dz;
        double exp_y0dy, exp_z0z0, exp_z0dz;

        // FIXME: consider the periodicity for [nx0:nx1]
        for (ix = nx0; ix < nx1; ix++) {
                x1xij = x0xij + ix*dx;
                tmpx = x1xij * aa_xx + y0yij * aa_xy + z0zij * aa_xz;
                tmpy = x1xij * aa_xy + y0yij * aa_yy + z0zij * aa_yz;
                tmpz = x1xij * aa_xz + y0yij * aa_yz + z0zij * aa_zz;
                _xyz0xyz0 = -x1xij * tmpx - y0yij * tmpy - z0zij * tmpz;
                if (_xyz0xyz0 < EXPMIN) {
// _xyz0dy (and _xyz0dz) can be very big, even greater than the effective range
// of exp function (and produce inf).  When exp_xyz0xyz0 is 0 and exp_xyz0dy is
// inf, the product will be ill-defined.  |_xyz0dy| should be smaller than
// |_xyz0xyz0| in any situations.  exp_xyz0xyz0 should dominate the product
// exp_xyz0xyz0 * exp_xyz0dy.  When exp_xyz0xyz0 is 0, the product should be 0.
// All the rest exp products should be smaller than exp_xyz0xyz0 and can be
// neglected.
                        pweights = weight_x + (ix-nx0)*l1l1;
                        for (n = 0; n < l1l1; n++) {
                                pweights[n] = 0;
                        }
                        continue;
                }
                _xyz0dy = -2 * dy * tmpy;
                _xyz0dz = -2 * dz * tmpz;
                exp_xyz0xyz0 = fac * exp(_xyz0xyz0);
                exp_xyz0dz = exp(_xyz0dz);

                //exp_xyz0dy = exp(_xyz0dy);
                //exp_y0dy = exp_xyz0dy * exp_dydy;
                exp_y0dy = exp(_xyz0dy + _dydy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                _z0dz = _xyz0dz;
                for (iy = grid_close_to_yij; iy < ny1; iy++) {
                        //pweights = weights + (ix * mesh[1] + iy) * mesh[2];
                        //exp_z0z0p = exp_z0z0;
                        //exp_z0dzp = exp_z0dz * exp_dzdz;
                        //for (iz = grid_close_to_zij; iz < mesh[2]; iz++) {
                        //        val += pweights[iz] * exp_z0z0p;
                        //        exp_z0z0p *= exp_z0dzp;
                        //        exp_z0dzp *= exp_2dzdz;
                        //}
                        //exp_z0z0p = exp_z0z0;
                        //exp_z0dzp = exp_dzdz / exp_z0dz;
                        //for (iz = grid_close_to_zij-1; iz >= 0; iz--) {
                        //        exp_z0z0p *= exp_z0dzp;
                        //        exp_z0dzp *= exp_2dzdz;
                        //        val += pweights[iz] * exp_z0z0p;
                        //}
                        pweights = weights + (ix * mesh[1] + iy) * mesh[2]; // FIXME ix -> mod(ix,mesh[0]) for periodicity
                        _nonorth_dot_z(weight_yz+(iy-ny0)*ngridz, pweights,
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz, _z0dz, _dzdz);
                        _z0dz += _dydz;
                        exp_z0z0 *= exp_y0dy;
                        exp_z0dz *= exp_dydz;
                        exp_y0dy *= exp_2dydy;
                }

                exp_y0dy = exp(_dydy - _xyz0dy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                _z0dz = _xyz0dz;
                for (iy = grid_close_to_yij-1; iy >= ny0; iy--) {
                        exp_z0z0 *= exp_y0dy;
                        exp_y0dy *= exp_2dydy;
                        _z0dz -= _dydz;
                        if (exp_dydz != 0) {
                                exp_z0dz /= exp_dydz;
                        } else {
                                exp_z0dz = exp(_z0dz);
                        }
                        pweights = weights + (ix * mesh[1] + iy) * mesh[2];
                        _nonorth_dot_z(weight_yz+(iy-ny0)*ngridz, pweights,
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz, _z0dz, _dzdz);
                }

                dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, &ngridy,
                       &D1, weight_yz, &ngridz, ys_exp, &ngridy,
                       &D0, weight_z, &ngridz);
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1, &ngridz,
                       &D1, zs_exp, &ngridz, weight_z, &ngridz,
                       &D0, weight_x+(ix-nx0)*l1l1, &l1);
        }
        dgemm_(&TRANS_N, &TRANS_N, &l1l1, &l1, &ngridx,
               &D1, weight_x, &l1l1, xs_exp, &ngridx,
               &D0, out, &l1l1);

        double *buf = out + l1l1l1;
        cache = buf + _MAX_RR_SIZE[topl];
        _affine_trans(buf, out, a, floorl, topl, cache);
        _plain_vrr2d(out, buf, cache, li, lj, ri, rj);

        int di = _LEN_CART[li];
        int dj = _LEN_CART[lj];
        int naoi = dims[0];
        int i, j;
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                mat[j*naoi+i] += out[j*di+i];
        } }
}

static int _nonorth_cache_size(int *mesh, int l)
{
        int dcart = _LEN_CART[l];
        int deriv = 1;
        int topl = l + l + deriv;
        int l1 = topl + 1;
        const int nimgs = 1;
        int cache_size = 0;
        cache_size += l1 * (mesh[0] + mesh[1] + mesh[2]) * nimgs;
        cache_size += mesh[1] * mesh[2]; // * nimgs * nimgs
        cache_size += l1 * mesh[2] * nimgs;
        cache_size += l1 * l1 * mesh[0];
        cache_size = MAX(cache_size, _MAX_AFFINE_SIZE[topl]*2);
        cache_size += l1 * l1 * l1;
        cache_size += _MAX_RR_SIZE[topl];
        return dcart*dcart + cache_size;
}

static int _max_cache_size(int (*fsize)(), int *shls_slice, int *bas, int *mesh)
{
        int i, n;
        int i0 = MIN(shls_slice[0], shls_slice[2]);
        int i1 = MAX(shls_slice[1], shls_slice[3]);
        int cache_size = 0;
        for (i = i0; i < i1; i++) {
                n = (*fsize)(mesh, bas(ANG_OF, i));
                cache_size = MAX(cache_size, n);
        }
        return cache_size+1000000;
}

// F_mat needs to be initialized as 0
void NUMINT_fill2c(void (*eval_mat)(), double *F_mat,
                   int comp, int hermi, int *shls_slice, int *ao_loc,
//?double complex *out, int nkpts, int comp, int nimgs,
//?double *Ls, double complex *expkL,
                   double log_prec, int dimension,
                   double *a, double *b, int *mesh, double *weights,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int naoi = ao_loc[ish1] - ao_loc[ish0];
        const int naoj = ao_loc[jsh1] - ao_loc[jsh0];
        //const int cache_size = _max_cache_size(_orth_cache_size, shls_slice,
        //                                       bas, mesh);
        const int cache_size = _max_cache_size(_nonorth_cache_size, shls_slice,
                                               bas, mesh);
#pragma omp parallel default(none) \
        shared(eval_mat, F_mat, comp, hermi, ao_loc, \
               log_prec, dimension, a, b, mesh, weights, \
               atm, natm, bas, nbas, env)
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0;
        int shls[2];
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*eval_mat)(F_mat+j0*naoi+i0, dims, 1., log_prec, dimension,
                            a, b, mesh, weights, shls, atm, bas, env, cache);
        }
        free(cache);
}
        if (hermi != PLAIN) { // lower triangle of F-array
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, F_mat+ic*naoi*naoi, hermi);
                }
        }
}


/*************************************************
 *
 * rho
 *
 *************************************************/
void GTOreverse_vrr2d_ket(double *g00, double *g01,
                          int li, int lj, double *ri, double *rj);

static void _cart_to_xyz(double *dm_xyz, double *dm_cart, int floorl, int topl)
{
        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int l, lx, ly, lz, n;
        for (n = 0; n < l1 * l1l1; n++) {
                dm_xyz[n] = 0;
        }

        for (n = 0, l = floorl; l <= topl; l++) {
        for (lx = l; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                dm_xyz[lx*l1l1+ly*l1+lz] += dm_cart[n];
        } } }
}

static void _dm_vrr6d(double *dm_cart, double *dm, int naoi,
                      int li, int lj, double *ri, double *rj, double *cache)
{
        int di = _LEN_CART[li];
        int dj = _LEN_CART[lj];
        double *dm_6d = cache;
        int i, j;
        for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        dm_6d[j*di+i] = dm[j*naoi+i];
                }
        }
        GTOreverse_vrr2d_ket(dm_cart, dm_6d, li, lj, ri, rj);
}

void NUMINTrho_3d_orth(double *rho, int *offset, int *ngrids,
                       double *dm, int naoi, int li, int lj,
                       double ai, double aj, double *ri, double *rj,
                       double fac, double log_prec,
                       int dimension, double *a, double *b, int *mesh,
                       double *cache)
{
        int topl = li + lj;
        double aij = ai + aj;
        double cutoff = gto_rcut(aij, topl, fac, log_prec);

        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        double *xs_exp = cache;
        double *ys_exp = xs_exp + l1 * mesh[0];
        double *zs_exp = ys_exp + l1 * mesh[1];
        cache = zs_exp + l1 * mesh[2];

        int img_slice[6];
        int grid_slice[6];
        _orth_components(xs_exp, img_slice, grid_slice, a[0], b[0], cutoff,
                         ri[0], rj[0], ai, aj, (dimension>=1), mesh[0], topl, cache);
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgx = nimgx1 - nimgx0;
        int nx0 = MAX(grid_slice[0], offset[0]);
        int nx1 = MIN(grid_slice[1], offset[0]+ngrids[0]);
        int ngridx = nx1 - nx0;
        if (nimgx == 1 && ngridx == 0) {
                return;
        }

        _orth_components(ys_exp, img_slice+2, grid_slice+2, a[4], b[4], cutoff,
                         ri[1], rj[1], ai, aj, (dimension>=2), mesh[1], topl, cache);
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgy = nimgy1 - nimgy0;
        int ny0 = MAX(grid_slice[2], offset[1]);
        int ny1 = MIN(grid_slice[3], offset[1]+ngrids[1]);
        int ngridy = ny1 - ny0;
        if (nimgy == 1 && ngridy == 0) {
                return;
        }

        _orth_components(zs_exp, img_slice+4, grid_slice+4, a[8], b[8], cutoff,
                         ri[2], rj[2], ai, aj, (dimension>=3), mesh[2], topl, cache);
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgz = nimgz1 - nimgz0;
        int nz0 = MAX(grid_slice[4], offset[2]);
        int nz1 = MIN(grid_slice[5], offset[2]+ngrids[2]);
        int ngridz = nz1 - nz0;
        if (nimgz == 1 && ngridz == 0) {
                return;
        }

        double *dm_xyz = cache;
        cache += l1l1 * l1;
        double *dm_cart = cache;
        _dm_vrr6d(dm_cart, dm, naoi, li, lj, ri, rj, dm_cart+_MAX_RR_SIZE[topl]);
        _cart_to_xyz(dm_xyz, dm_cart, li, topl);

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        int xcols = ngrids[1] * ngrids[2];
        double *xyr = cache;
        double *xqr = xyr + l1l1 * ngrids[2];
        int l;

        dgemm_(&TRANS_N, &TRANS_N, ngrids+2, &l1l1, &l1,
               &fac, zs_exp+offset[2], mesh+2, dm_xyz, &l1,
               &D0, xyr, ngrids+2);

//        if (nimgy == 1) {
//                for (l = 0; l <= topl; l++) {
//                        for (i = 0; i < (ny0-offset[1])*ngrids[2]; i++) {
//                                xqr[l*xcols+i] = 0;
//                        }
//                        for (i = (ny1-offset[1])*ngrids[2]; i < xcols; i++) {
//                                xqr[l*xcols+i] = 0;
//                        }
//                        dgemm_(&TRANS_N, &TRANS_T, ngrids+2, &ngridy, &l1,
//                               &D1, xyr+l*l1*ngrids[2], ngrids+2, ys_exp+ny0, mesh+1,
//                               &D0, xqr+l*xcols+(ny0-offset[1])*ngrids[2], ngrids+2);
//                }
//        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
//                for (l = 0; l <= topl; l++) {
//                        ngridy = ny1 - offset[1];
//                        dgemm_(&TRANS_N, &TRANS_T, ngrids+2, &ngridy, &l1,
//                               &D1, xyr+l*l1*ngrids[2], ngrids+2, ys_exp+offset[1], mesh+1,
//                               &D0, xqr+l*xcols, ngrids+2);
//                        for (i = (ny1-offset[1])*ngrids[2]; i < (ny0-offset[1])*ngrids[2]; i++) {
//                                xqr[l*xcols+i] = 0;
//                        }
//                        ngridy = offset[1] + ngrids[1] - ny0;
//                        dgemm_(&TRANS_N, &TRANS_T, ngrids+2, &ngridy, &l1,
//                               &D1, xyr+l*l1*ngrids[2], ngrids+2, ys_exp+ny0, mesh+1,
//                               &D0, xqr+l*xcols+(ny0-offset[1])*ngrids[2], ngrids+2);
//                }
//        } else {
                for (l = 0; l <= topl; l++) {
                        dgemm_(&TRANS_N, &TRANS_T, ngrids+2, ngrids+1, &l1,
                               &D1, xyr+l*l1*ngrids[2], ngrids+2, ys_exp+offset[1], mesh+1,
                               &D0, xqr+l*xcols, ngrids+2);
                }
//        }

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D1, rho+(nx0-offset[0])*xcols, &xcols);
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                ngridz = nz1 - offset[2];
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridy, &l1,
                       &D1, xqr, &xcols, xs_exp+offset[0], mesh,
                       &D1, rho, &xcols);
                ngridx = offset[0] + ngrids[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D1, rho+(nx0-offset[0])*xcols, &xcols);
        } else {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, ngrids, &l1,
                       &D1, xqr, &xcols, xs_exp+offset[0], mesh,
                       &D1, rho, &xcols);
        }
}

static void _nonorth_rho_z(double *rho, double *rhoz, int offset,
                           int nz0, int nz1, int grid_close_to_zij,
                           double e_z0z0, double e_z0dz, double e_dzdz,
                           double _z0dz, double _dzdz)
{
        if (e_z0z0 == 0) {
                return;
        }

        double exp_2dzdz = e_dzdz * e_dzdz;
        double exp_z0z0, exp_z0dz;
        int iz;

        exp_z0z0 = e_z0z0;
        exp_z0dz = e_z0dz * e_dzdz;
        for (iz = grid_close_to_zij; iz < nz1; iz++) {
                rho[iz-offset] += rhoz[iz-nz0] * exp_z0z0;
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
        }

        exp_z0z0 = e_z0z0;
        if (e_z0dz != 0) {
                exp_z0dz = e_dzdz / e_z0dz;
        } else {
                exp_z0dz = exp(_dzdz - _z0dz);
        }
        for (iz = grid_close_to_zij-1; iz >= nz0; iz--) {
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
                rho[iz-offset] += rhoz[iz-nz0] * exp_z0z0;
        }
}

void NUMINTrho_3d_nonorth(double *rho, int *offset, int *ngrids,
                          double *dm, int naoi, int li, int lj,
                          double ai, double aj, double *ri, double *rj,
                          double fac, double log_prec,
                          int dimension, double *a, double *b, int *mesh,
                          double *cache)
{
//FIXME: periodic condition
        double aij = ai + aj;
        double rij[3];
        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
        // rij_frac = einsum('ij,j->ik', b, rij)
        double xij_frac = rij[0] * b[0] + rij[1] * b[1] + rij[2] * b[2];
        double yij_frac = rij[0] * b[3] + rij[1] * b[4] + rij[2] * b[5];
        double zij_frac = rij[0] * b[6] + rij[1] * b[7] + rij[2] * b[8];
        double xi_frac = ri[0] * b[0] + ri[1] * b[1] + ri[2] * b[2];
        double yi_frac = ri[0] * b[3] + ri[1] * b[4] + ri[2] * b[5];
        double zi_frac = ri[0] * b[6] + ri[1] * b[7] + ri[2] * b[8];

        int floorl = li;
        int topl = li + lj;
        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        double xheights_inv = sqrt(SQUARE(b  ));
        double yheights_inv = sqrt(SQUARE(b+3));
        double zheights_inv = sqrt(SQUARE(b+6));

        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;
        xs_exp = cache;
        _nonorth_components(xs_exp, img_slice, grid_slice, (dimension>=1),
                            mesh[0], topl, offset[0], ngrids[0],
                            xi_frac, xij_frac, cutoff, xheights_inv);
        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ngridx = nx1 - nx0;
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgx = nimgx1 - nimgx0;
        if (nimgx == 1 && ngridx == 0) {
                return;
        }

        ys_exp = xs_exp + l1 * ngridx;
        _nonorth_components(ys_exp, img_slice+2, grid_slice+2, (dimension>=2),
                            mesh[1], topl, offset[1], ngrids[1],
                            yi_frac, yij_frac, cutoff, yheights_inv);
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int ngridy = ny1 - ny0;
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgy = nimgy1 - nimgy0;
        if (nimgy == 1 && ngridy == 0) {
                return;
        }

        zs_exp = ys_exp + l1 * ngridy;
        _nonorth_components(zs_exp, img_slice+4, grid_slice+4, (dimension>=3),
                            mesh[2], topl, offset[2], ngrids[2],
                            zi_frac, zij_frac, cutoff, zheights_inv);
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridz = nz1 - nz0;
        cache = zs_exp + l1 * ngridz;
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgz = nimgz1 - nimgz0;
        if (nimgz == 1 && ngridz == 0) {
                return;
        }

        double *dm_xyz = cache;
        cache += l1l1 * l1;
        double *dm_cart = cache;
        _dm_vrr6d(dm_cart, dm, naoi, li, lj, ri, rj, dm_cart+_MAX_RR_SIZE[topl]);
        _reverse_affine_trans(dm_xyz, dm_cart, a, floorl, topl,
                              dm_cart+_CUM_LEN_CART[topl]);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int inc1 = 1;
        // aa = einsum('ij,kj->ik', a, a)
        //double aa[9];
        //int n3 = 3;
        //dgemm_(&TRANS_T, &TRANS_N, &n3, &n3, &n3,
        //       &aij, a, &n3, a, &n3, &D0, aa, &n3);
        double aa_xx = aij * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double aa_xy = aij * (a[0] * a[3] + a[1] * a[4] + a[2] * a[5]);
        double aa_xz = aij * (a[0] * a[6] + a[1] * a[7] + a[2] * a[8]);
        double aa_yy = aij * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]);
        double aa_yz = aij * (a[3] * a[6] + a[4] * a[7] + a[5] * a[8]);
        double aa_zz = aij * (a[6] * a[6] + a[7] * a[7] + a[8] * a[8]);

        int ix, iy;
        double dx = 1. / mesh[0];
        double dy = 1. / mesh[1];
        double dz = 1. / mesh[2];

        //int grid_close_to_xij = rint(xij_frac * mesh[0]);
        int grid_close_to_yij = rint(yij_frac * mesh[1]);
        int grid_close_to_zij = rint(zij_frac * mesh[2]);
        //if (dimension < 1) {
        //        grid_close_to_xij = MIN(grid_close_to_xij, mesh[0]);
        //        grid_close_to_xij = MAX(grid_close_to_xij, 0);
        //}
        if (dimension < 2) {
                grid_close_to_yij = MIN(grid_close_to_yij, mesh[1]);
                grid_close_to_yij = MAX(grid_close_to_yij, 0);
        }
        if (dimension < 3) {
                grid_close_to_zij = MIN(grid_close_to_zij, mesh[2]);
                grid_close_to_zij = MAX(grid_close_to_zij, 0);
        }

        double img0_x = 0;
        double img0_y = 0;
        double img0_z = 0;
        double base_x = img0_x;// + dx * grid_close_to_xij;
        double base_y = img0_y + dy * grid_close_to_yij;
        double base_z = img0_z + dz * grid_close_to_zij;
        double x0xij = base_x - xij_frac;
        double y0yij = base_y - yij_frac;
        double z0zij = base_z - zij_frac;

        double _dydy = -dy * dy * aa_yy;
        double _dzdz = -dz * dz * aa_zz;
        double _dydz = -dy * dz * aa_yz * 2;
        double exp_dydy = exp(_dydy);
        double exp_2dydy = exp_dydy * exp_dydy;
        double exp_dzdz = exp(_dzdz);
        double exp_dydz = exp(_dydz);
        double x1xij, tmpx, tmpy, tmpz;
        double _xyz0xyz0, _xyz0dy, _xyz0dz, _z0dz;
        double exp_xyz0xyz0, exp_xyz0dz;
        double exp_y0dy, exp_z0z0, exp_z0dz;

        int xcols = ngridy * ngridz;
        double *xyr = cache;
        double *xqr = xyr + l1l1 * ngridz;
        double *rhoz = xqr + l1 * ngridy * ngridz;
        double *prho;
        int l;

        dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1l1, &l1,
               &D1, zs_exp, &ngridz, dm_xyz, &l1, &D0, xyr, &ngridz);

        for (l = 0; l <= topl; l++) {
                dgemm_(&TRANS_N, &TRANS_T, &ngridz, &ngridy, &l1,
                       &D1, xyr+l*l1*ngridz, &ngridz, ys_exp, &ngridy,
                       &D0, xqr+l*xcols, &ngridz);
        }

        // FIXME: consider the periodicity for [nx0:nx1]
        for (ix = nx0; ix < nx1; ix++) {
                x1xij = x0xij + ix*dx;
                tmpx = x1xij * aa_xx + y0yij * aa_xy + z0zij * aa_xz;
                tmpy = x1xij * aa_xy + y0yij * aa_yy + z0zij * aa_yz;
                tmpz = x1xij * aa_xz + y0yij * aa_yz + z0zij * aa_zz;
                _xyz0xyz0 = -x1xij * tmpx - y0yij * tmpy - z0zij * tmpz;
                if (_xyz0xyz0 < EXPMIN) {
                        continue;
                }
                _xyz0dy = -2 * dy * tmpy;
                _xyz0dz = -2 * dz * tmpz;
                exp_xyz0xyz0 = fac * exp(_xyz0xyz0);
                exp_xyz0dz = exp(_xyz0dz);

                //exp_xyz0dy = exp(_xyz0dy);
                //exp_y0dy = exp_xyz0dy * exp_dydy;
                exp_y0dy = exp(_xyz0dy + _dydy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                _z0dz = _xyz0dz;
                for (iy = grid_close_to_yij; iy < ny1; iy++) {
                        if (exp_z0z0 == 0) {
                                break;
                        }
                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, &inc1, &l1,
                               &D1, xqr+(iy-ny0)*ngridz, &xcols, xs_exp+ix-nx0, &ngridx,
                               &D0, rhoz, &ngridz);
                        prho = rho + ((ix-offset[0])*ngrids[1] + iy-offset[1]) * ngrids[2];
                        _nonorth_rho_z(prho, rhoz, offset[2],
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz, _z0dz, _dzdz);
                        _z0dz += _dydz;
                        exp_z0z0 *= exp_y0dy;
                        exp_z0dz *= exp_dydz;
                        exp_y0dy *= exp_2dydy;
                }

                exp_y0dy = exp(_dydy - _xyz0dy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                _z0dz = _xyz0dz;
                for (iy = grid_close_to_yij-1; iy >= ny0; iy--) {
                        exp_z0z0 *= exp_y0dy;
                        if (exp_z0z0 == 0) {
                                break;
                        }
                        _z0dz -= _dydz;
                        if (exp_dydz != 0) {
                                exp_z0dz /= exp_dydz;
                        } else {
                                exp_z0dz = exp(_z0dz);
                        }
                        exp_y0dy *= exp_2dydy;

                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, &inc1, &l1,
                               &D1, xqr+(iy-ny0)*ngridz, &xcols, xs_exp+ix-nx0, &ngridx,
                               &D0, rhoz, &ngridz);
                        prho = rho + ((ix-offset[0])*ngrids[1] + iy-offset[1]) * ngrids[2];
                        _nonorth_rho_z(prho, rhoz, offset[2],
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz, _z0dz, _dzdz);
                }
        }
}

static void _rho_loop(double *rho, int *offset, int *ngrids,
                      double *dm, int *dims, int *shls,
                      void (*eval_rho_3d)(), double log_prec,
                      int dimension, double *a, double *b, int *mesh,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      double *cache)
{
        const int naoi = dims[0];
        const int i_sh = shls[0];
        const int j_sh = shls[1];
        const int li = bas(ANG_OF, i_sh);
        const int lj = bas(ANG_OF, j_sh);
        double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        double ai = env[bas(PTR_EXP, i_sh)];
        double aj = env[bas(PTR_EXP, j_sh)];
        double ci = env[bas(PTR_COEFF, i_sh)];
        double cj = env[bas(PTR_COEFF, j_sh)];
        double aij = ai + aj;
        double eij, fac;
        double rrij = CINTsquare_dist(ri, rj);

        aij = ai + aj;
        eij = (ai * aj / aij) * rrij;
        fac = exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < 1e-40) {
                return;
        }

        (*eval_rho_3d)(rho, offset, ngrids, dm, naoi, li, lj, ai, aj, ri, rj,
                       fac, log_prec, dimension, a, b, mesh, cache);
}

static int _rho_cache_size(int l, int comp, int *ngrids)
{
        int l1 = l * 2 + 1;
        int cache_size = 0;
        cache_size += l1 * ngrids[1] * ngrids[2];
        cache_size += l1 * l1 * ngrids[2] * 2;
        cache_size = MAX(cache_size, 3*_MAX_RR_SIZE[l*2]);
        cache_size = MAX(cache_size, _CUM_LEN_CART[l*2]+2*_MAX_AFFINE_SIZE[l*2]);
        cache_size += l1 * (ngrids[0] + ngrids[1] + ngrids[2]);
        cache_size += l1 * l1 * l1;
        return cache_size;
}

/*
 * F_dm are a set of uncontracted cartesian density matrices
 */
void NUMINT_rho_drv(void (*eval_rho_3d)(), double *rho, int *offset, int *ngrids,
                    double *F_dm, int comp, int hermi, int *shls_slice, int *ao_loc,
//?double complex *out, int nkpts, int comp, int nimgs,
//?double *Ls, double complex *expkL,
                    double log_prec, int dimension,
                    double *a, double *b, int *mesh,
                    int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int naoi = ao_loc[ish1] - ao_loc[ish0];
        const int naoj = ao_loc[jsh1] - ao_loc[jsh0];

        int lmax = 0;
        int ib;
        for (ib = 0; ib < nbas; ib++) {
                lmax = MAX(lmax, bas(ANG_OF, ib));
        }
        const int cache_size = _rho_cache_size(lmax, comp, ngrids);

        double *rhobufs[MAX_THREADS];
#pragma omp parallel default(none) \
        shared(eval_rho_3d, rho, offset, ngrids, F_dm, comp, hermi, ao_loc, \
               log_prec, dimension, a, b, mesh, rhobufs, \
               atm, natm, bas, nbas, env)
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0, i, j, k;
        int shls[2];
        double *cache = malloc(sizeof(double) * cache_size);
        double *rho_priv = calloc(ngrids[0]*ngrids[1]*ngrids[2], sizeof(double));
        rhobufs[omp_get_thread_num()] = rho_priv;

#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                _rho_loop(rho_priv, offset, ngrids,
                          F_dm+j0*naoi+i0, dims, shls, eval_rho_3d,
                          log_prec, dimension, a, b, mesh,
                          atm, natm, bas, nbas, env, cache);
        }
        NPomp_dsum_reduce_inplace(rhobufs, ngrids[0]*ngrids[1]*ngrids[2]);
#pragma omp master
        {
                double *p0, *p1;
                rho += (offset[0] * mesh[1] + offset[1]) * mesh[2] + offset[2];
                for (i = 0; i < ngrids[0]; i++) {
                for (j = 0; j < ngrids[1]; j++) {
                        p1 = rho + (i * mesh[1] + j) * mesh[2];
                        p0 = rho_priv + (i * ngrids[1] + j) * ngrids[2];
                        for (k = 0; k < ngrids[2]; k++) {
                                p1[k] += p0[k];
                        }
                } }
        }
        free(cache);
        free(rho_priv);
}
}