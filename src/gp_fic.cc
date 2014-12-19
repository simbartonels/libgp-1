// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_fic.h"
#include "cov_factory.h"
#include "basis_functions/basisf_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
const double log2pi = log(2 * M_PI);

//TODO: find a way to get the value from the super class
const double initial_L_size = 1000;

FICGaussianProcess::FICGaussianProcess(size_t input_dim, std::string covf_def,
		size_t num_basisf, std::string basisf_def) :
		AbstractGaussianProcess(input_dim, covf_def) {
	BasisFFactory factory;
	//wrap initialized covariance function with basis function
	cf = factory.createBasisFunction(basisf_def, num_basisf, cf);
	cf->loghyper_changed = 0;
	bf = (IBasisFunction *) cf;
	M = bf->getNumberOfBasisFunctions();
	alpha.resize(M);
	//TODO: are all 3 matrices necessary?
	L.resize(M, M);
	Lu.resize(M, M);
	Luu.resize(M, M);
	W.resize(M, M);
	BWdg.resize(M, M);
	w.resize(M);
	beta.resize(M);
	k_star.resize(M);
	dKuui.resize(M, M);
	temp.resize(M);
}

FICGaussianProcess::~FICGaussianProcess() {
}

double FICGaussianProcess::var_impl(const Eigen::VectorXd x_star) {
	//TODO: as far a I can tell this is the only usage of L
	//=> it's probably sufficient to really use only the Cholesky
	//and not necessary to compute the inverse
	return bf->getWrappedKernelValue(x_star, x_star)
			+ k_star.transpose() * L * k_star;
}

void FICGaussianProcess::computeCholesky() {
	//TODO: refactor! this method is too long!
	/*
	 * This method does not compute the Cholesky in the same sense as
	 * the GaussianProcess class does. Here the same thing happens as
	 * in infFITC.m from the gpml toolbox by Rasmussen and Nikisch. The
	 * method name computeCholesky is kept for abstraction reasons.
	 */
	size_t n = sampleset->size();

	if (n > isqrtgamma.rows()) {
		k.resize(n);
		isqrtgamma.resize(n);
		dg.resize(n);
		V.resize(M, n);
		Phi.resize(M, n);
	}
	for (size_t i = 0; i < n; i++) {
		Phi.col(i) = bf->computeBasisFunctionVector(sampleset->x(i));
		k(i) = bf->getWrappedKernelValue(sampleset->x(i), sampleset->x(i));
	}
	Luu = bf->getCholeskyOfInvertedSigma();
	/*
	 * TODO: could we just multiply Phi with sqrt(gamma) HERE instead of using
	 * the inverse later? What's more stable?
	 */
	V = Luu.topLeftCorner(M, M).triangularView<Eigen::Lower>().solve(Phi);
	//noise is already added in k
	dg = k - (V.transpose() * V).diagonal();
//	isqrtgamma = isqrtgamma.cwiseInverse().sqrt();
	isqrtgamma.array() = 1 / dg.array().sqrt();
	V = V * isqrtgamma.asDiagonal();
	// TODO: V*V^T is symmetric and it should be possible to reduce the costs.
	//Especially since V is an Mxn matrix!
	Lu = V * V.transpose() + Eigen::MatrixXd::Identity(M, M);
	Lu.topLeftCorner(M, M) = Lu.llt().matrixL();

	/*
	 * Here we have to divert from the Matlab implementation. in Matlab all
	 * the matrices are upper matrices. Here we have what they are supposed to be:
	 * lower matrices.
	 */
	//TODO: avoid allocation
	Eigen::MatrixXd temp = Luu * Lu;
	//the line below does not work. why?
//	L = L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(M, M));
	L = temp.triangularView<Eigen::Lower>().solve(
			Eigen::MatrixXd::Identity(M, M));
	temp.transpose().triangularView<Eigen::Upper>().solveInPlace(L);
	L = L - bf->getSigma();
}

void FICGaussianProcess::updateCholesky(const double x[], double y) {
	//Do nothing and just recompute everything.
	//TODO: might be a slow down in applications!
	cf->loghyper_changed = true;
}

void FICGaussianProcess::update_k_star(const Eigen::VectorXd &x_star) {
	k_star = bf->computeBasisFunctionVector(x_star);
}

void FICGaussianProcess::update_alpha() {
	size_t n = sampleset->size();
	if (n > r.size()) {
		r.resize(n);
	}
	// Map target values to VectorXd
	const std::vector<double>& targets = sampleset->y();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	r.array() = y.array() * isqrtgamma.array();
	beta = V * r;
	/*
	 * In the Matlab implementation Luu and Lu are upper matrices and that's
	 * why we need to transpose here.
	 */
	Lu.triangularView<Eigen::Lower>().solveInPlace(beta);
	//alpha = Luu\(Lu\be)
	alpha = Lu.transpose().triangularView<Eigen::Upper>().solve(beta);
	Luu.transpose().triangularView<Eigen::Upper>().solveInPlace(alpha);
}

double FICGaussianProcess::log_likelihood_impl() {
	//TODO: move stuff that is parameter independent to computeAlpha or something
	double t = 0;
	double t3 = 0;
	//TODO: is this method numerically stable? what are typical values?
	for (size_t i = 0; i < M; i++) {
		t += log(Lu(i, i));
		t3 -= beta(i) * beta(i);
	}
	size_t n = sampleset->size();
	double t2 = 0;
	for (size_t i = 0; i < n; i++) {
		t2 += log(dg(i)) + r(i) * r(i);
	}
	//TODO: is this better? or should it be moved to the loop?
	t2 = t2 + n * log2pi;

	//TODO: the following call should be more efficient than the loops below. Does it compile?
//	t = Lu.diagonal().log().sum()-beta.squaredNorm()+dg.log().sum()+r.squaredNorm()+n*log2pi;
	return t + (t2 + t3) / 2;
}

Eigen::VectorXd FICGaussianProcess::log_likelihood_gradient_impl() {
	//TODO: move stuff that is parameter independent to computeAlpha or something
	//TODO: move allocations to constructor
	size_t num_params = bf->get_param_dim();
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(num_params);
	const std::vector<double>& targets = sampleset->y();
	size_t n = sampleset->size();
	Eigen::Map<const Eigen::VectorXd> y(&targets[0], n);
	//TODO: move to extra function
	if (n > al.size()) {
		al.resize(n);
		W.resize(M, n);
		Wdg.resize(M, n);
		B.resize(M, n);
		ddiagK.resize(n);
		dKui.resize(M, n);
		R.resize(M, n);
		v.resize(n);
		WdgSum.resize(n);
	}

	dKui.setZero();
	dKuui.setZero();
	ddiagK.setZero();

	//    W = Ku./repmat(sqrt(dg)',nu,1);
	//    W = chol(Kuu+W*W'+snu2*eye(nu))'\Ku;
	W =
			(Phi * dg.cwiseInverse().asDiagonal() * Phi.transpose()
					+ bf->getInverseOfSigma()).selfadjointView<Eigen::Lower>().llt().matrixL().solve(
					Phi);

	//    al = (y-m - W'*(W*((y-m)./dg)))./dg;
	al = (y - W.transpose() * (W * (y.cwiseQuotient(dg)))).cwiseQuotient(dg);
//    B = iKuu*Ku;
	B = bf->getSigma() * Phi;
//    % = Upsi^(-1)*Uvx
//    Wdg = W./repmat(dg',nu,1); w = B*al;
	Wdg = W * dg.cwiseInverse().asDiagonal();
	w = B * al;
	BWdg = B * Wdg.transpose();
	WdgSum = Wdg.array().square().matrix().colwise().sum();
	for (size_t i = 0; i < num_params; i++) {
//      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);
		bool gradiSigmaIsNull = bf->gradiSigmaIsNull(i);
		bool gradBasisFunctionIsNull = bf->gradBasisFunctionIsNull(i);
		if (!gradiSigmaIsNull) {
			bf->gradiSigma(i, dKuui);
		} else
			//TODO: this case occurs!!!
			dKuui.setZero();

		if (!gradBasisFunctionIsNull) {
			for (size_t j = 0; j < n; j++) {
				bf->gradBasisFunction(sampleset->x(j), Phi.col(j), i, temp);
				//TODO: is it possible to avoid the copy?
				dKui.col(j) = temp;
			}
			//R = 2*dKui-dKuui*B;
			R = 2 * dKui - dKuui * B;
		} else {
			R = -dKuui * B;
			//TODO: NO!
			dKui.setZero();
		}

		double ddiagK_idg;
		if (!bf->gradDiagWrappedIsNull(i)) {
			bf->gradDiagWrapped(sampleset, k, i, ddiagK);
			ddiagK_idg = ddiagK.cwiseQuotient(dg).sum();
			// v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
			v = ddiagK.transpose() - R.cwiseProduct(B).colwise().sum();
		} else {
			ddiagK_idg = 0;
			v = -R.cwiseProduct(B).colwise().sum();
		}

//      dnlZ.cov(i) = (ddiagKi'*(1./dg) +w'*(dKuui*w-2*(dKui*al)) -al'*(v.*al) ...
//                         - sum(Wdg.*Wdg,1)*v - sum(sum((R*Wdg').*(B*Wdg'))) )/2;
		gradient(i) = ddiagK_idg
				//TODO: line below can be optimized (if either dKuui or dKui are 0)
				+ (w.transpose() * (dKuui * w - 2 * (dKui * al))).sum()
				- (v.array() * al.array().square()).sum()
				- WdgSum.cwiseProduct(v).sum()
				- (R * Wdg.transpose()).cwiseProduct(BWdg).sum();
	}
	gradient /= 2;
	//noise gradient included in the loop above
	return gradient;
}
}
