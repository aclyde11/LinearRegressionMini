#include <iostream>

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include "Classifier.h"
#include <LinearRegression.h>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int check_correct_thetas(MatrixXd A, MatrixXd B, double epsilon) {
    return (B-A).squaredNorm() < epsilon;
}

int main(int argc, const char* argv[]) {
    Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]", "{", "}\n");

    MatrixXd X = load_csv<MatrixXd>(argv[1]);
    MatrixXd y = load_csv<MatrixXd>(argv[2]);
    MatrixXd theta_correct = load_csv<MatrixXd>(argv[3]);


    LinearRegression clf(atof(argv[4]), atof(argv[5]), 0.0001);
    clf.fit(X,y);

    std::cout << check_correct_thetas(clf.getThetas(), theta_correct, 0.001);

    return 0;
}


