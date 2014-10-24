/* solve_chol - solve a linear system A*X = B using the cholesky factorization
   of A (where A is square, symmetric and positive definite.

   Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-12-16. */

#include "mex.h"
#include <math.h>
#include <string.h>
#include "fic_gp.h"
#include <Eigen/Dense>
#include <iostream>

//extern int dpotrs_(char *, long *, long *, double *, long *, double *, long *, long *);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  size_t M;
  size_t D;
  size_t n;
  size_t p;
  if (nrhs != 4 || nlhs != 2)                              /* check the input */
    mexErrMsgTxt("Usage: [alpha, L] = infSMmex(M, unwrap(hyp), x, y)");
  M = (int) mxGetScalar(prhs[0]);
  n = mxGetN(prhs[2]);
  D = mxGetM(prhs[2]);
  p = mxGetN(prhs[1]);
  libgp::FICGaussianProcess gp(D, "CovSum ( CovSEiso, CovNoise)", M, "SparseMultiScaleGP");
//  Eigen::VectorXd params = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[1]), p);
//  gp.covf().set_loghyper(params);
//
//  Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(mxGetPr(prhs[3]), n);
//  Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(mxGetPr(prhs[1]), n, D);
//  for(size_t i; i < n; i++){
//	  gp.add_pattern(X.row(i), y(i));
//  }
  plhs[0] = mxCreateDoubleMatrix(M, n, mxREAL);  /* allocate space for output */
//  Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[0]), M, n) = gp.getAlpha();
  plhs[1] = mxCreateDoubleMatrix(M, M, mxREAL);
//  Eigen::Map<Eigen::MatrixXd>(mxGetPr(plhs[1]), M, M) = gp.getL();
}

