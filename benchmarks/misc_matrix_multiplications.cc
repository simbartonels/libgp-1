#include "util/time_call.h"
#include <Eigen/Dense>

Eigen::MatrixXd Mat1;
Eigen::MatrixXd Mat2;
Eigen::VectorXd v;
Eigen::VectorXd u;

void f_base() {
	double x = (v.transpose() * (Mat1 * u + Mat2 * v)).sum();
}

void f2(){
	double x = (v.transpose() * Mat1 * u + v.transpose() * Mat2 * v).sum();
}

void f3(){
	double x = (v.transpose() * Mat1 * u).sum();
	x += (v.transpose() * Mat2 * v).sum();
}

int main(int argc, char const *argv[]) {
	size_t msize = 4000;
	Mat1.resize(msize, msize);
	Mat1.setRandom();
	v.resize(msize);
	v.setRandom();
	Mat2.resize(msize, msize);
	Mat2.setRandom();
	u.resize(msize);
	u.setRandom();

	compare_time(f_base, f2, 20);
	compare_time(f_base, f3, 20);
}
