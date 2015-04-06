#include <cmath>

#include "gp_fic.h"
#include "gp_multiscale_optimized.h"
#include "gp_fic_optimized.h"
#include "naive/gp_fic_naive.h"
#include "gp_utils.h"
#include "util/time_call.h"

using namespace libgp;

    static Eigen::VectorXd static_test;

    void testing(const Eigen::VectorXd& v){
    	std::cout << "v data: " << v.data() << std::endl;
    }

    void testing2(Eigen::VectorXd& out){
    	out(0) = -1.;
    }

    void testing3(Eigen::MatrixXd& out){
    	out(0, 0) = -2.;
    	std::cout << "outMat data: " << out.data() << std::endl;
    	std::cout << "outMat ref: " << &out << std::endl;
    }

    void testPassingReferences(){
    	size_t input_dim = 3;
    	size_t M = 100;
    	size_t n = 5000;

    	//TODO: find a way to extract a vector from a matrix without copying
    	//TODO: then adapt SampleSet

    	Eigen::MatrixXd Mat(M, M);
    	Mat.setRandom();
    	Eigen::Map<const Eigen::VectorXd> vec(Mat.row(0).data(), M);
    //	std::cout << "vec " << vec.transpose() << std::endl;
    //	std::cout << "M.col(0) " << Mat.col(0).transpose() << std::endl;

    	Eigen::VectorXd copyvec = vec;
    	std::cout << "copyvec data: " << copyvec.data() << std::endl;

    	//TODO: calls like these trigger a copy!
    	std::cout << "Mat(0) data: " << Mat.row(0).data() << std::endl;
    	testing(Mat.row(0));

    	std::cout << "Mat(0) data: " << Mat.col(0).data() << std::endl;
    	testing(Mat.col(0));

    	std::cout << "vec data: " << vec.data() << std::endl;
    	testing(vec);

    	std::cout << "before: " << copyvec(0) << std::endl;
    	testing2(copyvec);
    	std::cout << "after: " << copyvec(0) << std::endl;
    	std::cout << "M after: " << Mat(0, 0) << std::endl;

    	Eigen::Map<Eigen::MatrixXd, Eigen::Aligned> MatMap(Mat.data(), M, M);
    	std::cout << "before: " << MatMap(0, 0) << std::endl;
    	std::cout << "Mat ref: " << &Mat << std::endl;
    	testing3(Mat);
    	std::cout << "after: " << MatMap(0, 0) << std::endl;
    	std::cout << "M after: " << Mat(0, 0) << std::endl;


    //	std::cout << "before: " << vec(0) << std::endl;
    //	testing2(vec);
    //	std::cout << "after: " << vec(0) << std::endl;
    //
    //	std::cout << "before: " << Mat(0, 0) << std::endl;
    //	testing2(Mat.col(0));
    //	std::cout << "after: " << Mat(0, 0) << std::endl;

//    	std::cout << "M before: " << Mat(0, 0) << std::endl;
//    	testing3(Mat.col(0));
//    	std::cout << "M after: " << Mat(0, 0) << std::endl;

    }

    Eigen::VectorXd returnVector(){
    	return static_test;
    }

    Eigen::VectorXd& returnVectorRef(){
    	return static_test;
    }

    const Eigen::VectorXd returnConstVector(){
    	return static_test;
    }

    const Eigen::VectorXd& returnConstVectorRef(){
    	return static_test;
    }

    void testCopying(){
    	static_test.resize(10);
    	static_test.setRandom();
    	//not allowed
//    	std::cout << "func ref: " << &(returnVector()) << std::endl;
    	Eigen::Map<Eigen::VectorXd> p(returnVector().data(), 10);
//    	std::cout << "func ref: " << p.data() << std::endl;
    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
    	p(0) = 0.;
    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	p = Eigen::Map<Eigen::VectorXd>(returnVectorRef().data(), 10);
    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
    	p(0) = 0.;
    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

//    	p = Eigen::Map<Eigen::VectorXd>(returnConstVector().data(), 10);
//    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
//    	p(0) = 0.;
//    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	std::cout << "vec ref: " << static_test.data() << std::endl;
    	std::cout << "func ref: " << returnConstVector().data() << std::endl;

//    	p = Eigen::Map<Eigen::VectorXd>(returnConstVectorRef().data(), 10);
//    	std::cout << "vec(0) before: " << static_test(0) << std::endl;
//    	p(0) = 0.;
//    	std::cout << "vec(0) after: " << static_test(0) << std::endl;

    	std::cout << "vec ref: " << static_test.data() << std::endl;
    	std::cout << "func ref: " << returnConstVectorRef().data() << std::endl;

    	const Eigen::VectorXd & constRef = returnConstVectorRef();
//    	double * ref = constRef.data();
//    	const Eigen::Map<Eigen::VectorXd> pconst(constRef.data(), 10);

    }

    static FICGaussianProcess * gp;

    static FICnaiveGaussianProcess * gpnaive;

    static FICGaussianProcess * gpopt;

    void f1(){
    	gp->log_likelihood();
    }

    void f2(){
    	gpnaive->log_likelihood();
    }

    void testSpeedOfLogLikelihood(){
    	double fv = gp->log_likelihood();
    	double nv = gpnaive->log_likelihood();
    //	std::cout << "fv: " << fv << std::endl;
    //	std::cout << "nv: " << nv << std::endl;
    	assert(std::abs(fv-nv)/nv <1e-5);

    	std::cout << "speed of llh computation:" << std::endl;
    	compare_time(f1, f2, 5);
    }

    static Eigen::MatrixXd V;
    static Eigen::MatrixXd Sigma1;
    static Eigen::MatrixXd Sigma2;
    static Eigen::MatrixXd Sigma3;
    static Eigen::VectorXd dg;

    void f1CholComp(){
//    	for(size_t i = 0; i < 100; i++)
    	dg = V.array().square().colwise().sum().transpose();
    }

    void f2CholComp(){
//    	for(size_t i = 0; i < 100; i++)
    	dg = (V.transpose() * V).diagonal();
    }

    void f3CholComp(){
//    	Sigma1.setZero();
    	Sigma1.selfadjointView<Eigen::Lower>().rankUpdate(V);
    	dg = Sigma1.diagonal();
    }


    void testSpeedOfCholeskyComputation1(){
    	compare_time(f1CholComp, f2CholComp, 100);
    }

    void testSpeedOfCholeskyComputation11(){
    	compare_time(f1CholComp, f3CholComp, 100);
    }

    void f1CholComp2(){
    	gp->covf().loghyper_changed = true;
    	gp->getL();
    }

    void f2CholComp2(){
    	gpnaive->covf().loghyper_changed = true;
    	gpnaive->getL();
    }

    void testSpeedOfCholeskyComputation2(){
    	compare_time(f1CholComp2, f2CholComp2, 10);
    }

    void llhGradBaseline(){
    	gpnaive->log_likelihood_gradient();
    }

    void llhGradFast(){
    	gp->log_likelihood_gradient();
    }

    void llhGradFaster(){
    	gpopt->log_likelihood_gradient();
    }

    void LuuLu_base(){
    	Sigma3 = Sigma1 * Sigma2;
    }

    void LuuLu_2(){
    	Sigma3 = Sigma1.triangularView<Eigen::Lower>() * Sigma2;
    }

    void testSpeedOfLuuLu(){
        	compare_time(LuuLu_base, LuuLu_2, 10);
        }

    static Eigen::VectorXd v;
    static Eigen::MatrixXd B;
    static Eigen::VectorXd dkuui;

    void m_base(){
    	size_t j = 1;
    	size_t m = 2;
    	v.array() += B.row(j).array() * dkuui(j) * B.row(m).array();
    }

    void m_competitor(){
    	size_t j = 1;
    	size_t m = 2;
    	v += B.row(j).cwiseProduct(dkuui(j) * B.row(m));
    }

    void compareMspeed(){
    	size_t n = 6000;
    	size_t M = 1500;
    	v.resize(n);
    	v.setRandom();
    	dkuui.resize(M);
    	dkuui.setRandom();
    	B.resize(M, n);
    	B.setRandom();
    	compare_time(m_base, m_competitor, 10);
    }

    Eigen::MatrixXd Delta;
    Eigen::VectorXd x;
    Eigen::MatrixXd U;
    Eigen::MatrixXd Uell;
    Eigen::VectorXd logfactors;
    Eigen::VectorXd uvx;
    void predict_base(){
    	size_t M = U.rows();
    	Delta = x.transpose().replicate(M, 1) - U;
    	//	Delta.array() = Delta.array().square() / Uell.array();
		uvx = (Delta.array().square() / Uell.array()).rowwise().sum();
		uvx.array() = (-0.5 * uvx.array() - logfactors.array()).exp();
    }

    void predict_competitor(){
    	//slower
    	size_t M = U.rows();
    	for(size_t i = 0; i < M; i++)
    		Delta.row(i) = x.transpose() - U.row(i);
		uvx = (Delta.array().square() / Uell.array()).rowwise().sum();
		uvx.array() = (-0.5 * uvx.array() - logfactors.array()).exp();
    }

    void compare_prediction(){
    	size_t n = 6000;
    	size_t M = 1500;
    	size_t D = 2;
    	Delta.resize(M, D);
    	Delta.setRandom();
    	x.resize(D);
    	x.setRandom();
    	U.resize(M, D);
    	U.setRandom();
    	Uell.resize(M, D);
    	Uell.setRandom();
    	logfactors.resize(M);
    	logfactors.setRandom();
    	uvx.resize(M);
    	uvx.setRandom();
    	compare_time(predict_base, predict_competitor, 10);
    }

    void measureBFcomputationTime() {
    	size_t num_execs = 100;

    	while (true) {
    		std::cout << "Choose M: ";
    		size_t M;
    		std::cin >> M;
		std::cout << "Choose D: ";
		size_t D;
		std::cin >> D;
		std::cout << "Choose n: ";
		size_t n;
		std::cin >> n;
    	Eigen::VectorXd grad(D);
    	Eigen::VectorXd x(D);
    	x.setRandom();
    	Eigen::MatrixXd X(n, D);
    	X.setRandom();
    	Eigen::VectorXd y(n);
    	y.setRandom();

    		std::cout << "initializing GP" << std::endl;
    		gp = new FICGaussianProcess(D, "CovSum ( CovSEard, CovNoise)", M,
    				"SparseMultiScaleGP");
    		for (int i = 0; i < n; ++i) {
    			gp->add_pattern(X.row(i), y(i));
    		}
    		gp->log_likelihood();
    		std::cout << "done" << std::endl;
		double min = -log(0.0);
		for(size_t j = 0; j < num_execs; j++){
    		stop_watch();
    		for (size_t i = 0; i < num_execs; i++) {
    			gp->f(x);
    			gp->var(x);
    		}
    		double tic = stop_watch() / num_execs;
		if(tic < min)
			min = tic;
    		std::cout << j << "/" << num_execs << ": " << min << std::endl;
		}
    	}
    }

    void measureLlhGradcomputationTime() {
    	while (true) {
    		std::cout << "Choose n: ";
			size_t n;
			std::cin >> n;
    		std::cout << "Choose M: ";
    		size_t M;
    		std::cin >> M;
		std::cout << "Choose D: ";
		size_t D;
		std::cin >> D;
		Eigen::VectorXd grad(D);
        	Eigen::VectorXd x(D);
        	x.setRandom();
        	Eigen::MatrixXd X(n, D);
        	X.setRandom();
        	Eigen::VectorXd y(n);
        	y.setRandom();
    		std::cout << "initializing GP" << std::endl;
//    		gp = new OptFICGaussianProcess(D, "CovSum ( CovSEard, CovNoise)", M,
//    				"FIC");
    		gpopt = new OptMultiScaleGaussianProcess(D, "CovSum ( CovSEard, CovNoise)", M,
    		    				"SparseMultiScaleGP");
    		for (int i = 0; i < n; ++i) {
//    			gp->add_pattern(X.row(i), y(i));
    			gpopt->add_pattern(X.row(i), y(i));
    		}
//    		gp->log_likelihood();
    		gpopt->log_likelihood();
    		std::cout << "done" << std::endl;
    		stop_watch();
//    		gp->log_likelihood_gradient();
    		double tic = stop_watch();
    		std::cout << "fic: " << tic << std::endl;

    		stop_watch();
    		gpopt->log_likelihood_gradient();
    		tic = stop_watch();
    		std::cout << "multiscale: " << tic << std::endl;
    	}
    }



int main(int argc, char const *argv[]) {
//	compareMspeed();
//	measureLlhGradcomputationTime();
//	measureBFcomputationTime();
//	compare_prediction();
//	return 0;
	size_t input_dim = 2;
	size_t M = 100;
	size_t n = 600;
	V.resize(M, n);
	V.setRandom();
	Sigma1.resize(M, M);
	Sigma2.resize(M, M);
	Sigma3.resize(M, M);
	Sigma1.setRandom();
	Sigma2.setRandom();
	Sigma3.setRandom();
//	gp = new FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
//			"SparseMultiScaleGP");
//	gpopt = new OptMultiScaleGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
//			"SparseMultiScaleGP");
	gp = new FICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
			"FIC");
	gpopt = new OptFICGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)", M,
			"FIC");
	gpnaive = new FICnaiveGaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)",
				M, "SparseMultiScaleGP");


	// initialize hyper parameter vector
	Eigen::VectorXd params(gp->covf().get_param_dim());
	params.setRandom();
	// set parameters of covariance function
	gp->covf().set_loghyper(params);
//	gpnaive->covf().set_loghyper(params);
	gpopt->covf().set_loghyper(params);
	// add training patterns
	for (int i = 0; i < n; ++i) {
		double x[] = { drand48() * 4 - 2, drand48() * 4 - 2 };
		double y = Utils::hill(x[0], x[1]) + Utils::randn() * 0.1;
		gp->add_pattern(x, y);
		gpnaive->add_pattern(x, y);
		gpopt->add_pattern(x, y);
	}

	Eigen::VectorXd gradnaive = gpnaive->log_likelihood_gradient();
	Eigen::VectorXd gradfaster = gpopt->log_likelihood_gradient();
	double diff = ((gradnaive.array() - gradfaster.array()).abs()/(gradnaive.array().abs()+1-50)).maxCoeff();
	if(diff >= 1e-5){
		std::cout << "correct gradient:" << std::endl << gradnaive.transpose() << std::endl;
		std::cout << "fast gradient: " << std::endl << gradfaster.transpose() << std::endl;
    	assert(diff < 1e-5);
	}
	std::cout << "starting speed comparison";
//	compare_time(llhGradBaseline, llhGradFast, 1);
	stop_watch();
	llhGradFaster();
	std::cout << "faster grad: " << stop_watch() << std::endl;
//	compare_time(llhGradFast, llhGradFaster, 1);

//	testSpeedOfLogLikelihood();
//	testSpeedOfCholeskyComputation2();


//	testSpeedOfCholeskyComputation11();
//	testSpeedOfLuuLu();
//	measureBFcomputationTime();

	delete gp;
	delete gpnaive;
	return 0;
}
