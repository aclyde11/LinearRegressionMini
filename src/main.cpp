#include <iostream>

#include <Eigen/Dense>
#include "Classifier.h"
#include <LinearRegression.h>

int main() {
    Eigen::MatrixXd m(5,2);
    m(0,0) = 1;
    m(1,0) = 1;
    m(2,0) = 1;
    m(3,0) = 1;
    m(4,0) = 1;
    m(0,1) = 12.4;
    m(1,1) = 14.3;
    m(2,1) = 14.5;
    m(3,1) = 14.9;
    m(4,1) = 16.1;

    Eigen::VectorXd v(5);
    v(0) = 11.2;
    v(1) = 12.5;
    v(2) = 12.7;
    v(3) = 13.1;
    v(4) = 14.1;

    LinearRegression clf(0.001, 100, 0.01);
    clf.fit(m,v);

    Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");
    std::cout << clf.getThetas().format(fmt);

    return 0;
}