#ifndef HUNGARIAN_HPP_INCLUDED
#define HUNGARIAN_HPP_INCLUDED

#include <Eigen/Dense>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXb;


void findMatching(Eigen::MatrixXd& m, Eigen::VectorXi& assignment);

void step2a(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, MatrixXb& starMatrix,
            MatrixXb& newStarMatrix, MatrixXb& primeMatrix, Eigen::VectorXi& coveredColumns,
            Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step2b(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, MatrixXb& starMatrix, 
            MatrixXb& newStarMatrix, MatrixXb& primeMatrix, Eigen::VectorXi& coveredColumns, 
            Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step3(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, MatrixXb& starMatrix, 
           MatrixXb& newStarMatrix, MatrixXb& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void step4(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, MatrixXb& starMatrix, 
           MatrixXb& newStarMatrix, MatrixXb& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);

void step5(Eigen::VectorXi& assignment, Eigen::MatrixXd& m, MatrixXb& starMatrix, 
           MatrixXb& newStarMatrix, MatrixXb& primeMatrix, Eigen::VectorXi& coveredColumns, 
           Eigen::VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim);

void buildassignmentvector(Eigen::VectorXi& assignment, MatrixXb& starMatrix, int nOfRows, int nOfColumns);

#endif
