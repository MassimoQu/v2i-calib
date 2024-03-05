#include "hungarian.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXi;


void findMatching(MatrixXd& m, VectorXi& assignment)
{
    int nOfRows = m.rows();
    int nOfColumns = m.cols();
    int minDim, row, col;

    double value, minValue;

    MatrixXb starMatrix(m.rows(), m.cols());
    MatrixXb newStarMatrix(m.rows(), m.cols());
    MatrixXb primeMatrix(m.rows(), m.cols());
    // VectorXi coveredColumns(m.rows());
    // VectorXi coveredRows(m.cols());
    VectorXi coveredColumns(m.cols());
    VectorXi coveredRows(m.rows());


    starMatrix.fill(0);
    newStarMatrix.fill(0);
    primeMatrix.fill(0);
    coveredColumns.fill(0);
    coveredRows.fill(0);

    assignment.fill(0);

    for (row = 0; row<nOfRows; row++)
    for (col = 0; col<nOfColumns; col++)
    if (m(row, col) < 0)
        throw std::invalid_argument("All matrix elements have to be positive!");

    /* preliminary steps */
    if (nOfRows <= nOfColumns)
    {
        minDim = nOfRows;

        for (row = 0; row<nOfRows; row++)
        {
            /* find the smallest element in the row */
            minValue = m(row, 0);
            for (col = 0; col<nOfColumns; col++)
            {
                value = m(row, col);
                if (value < minValue)
                    minValue = value;
            }

            /* subtract the smallest element from each element of the row */
            for (col = 0; col<nOfColumns; col++)
                m(row, col) -= minValue;
        }

        /* Steps 1 and 2a */
        for (row = 0; row<nOfRows; row++)
            for (col = 0; col<nOfColumns; col++)
                if (m(row, col) == 0)
                    if (!coveredColumns(col))
                    {
                        starMatrix(row, col) = true;
                        coveredColumns(col) = true;
                        break;
                    }
    }
    else /* if(nOfRows > nOfColumns) */
    {
        minDim = nOfColumns;

        for (col = 0; col<nOfColumns; col++)
        {
            /* find the smallest element in the column */
            minValue = m(0, col);
            for (row = 0; row<nOfRows; row++)
            {
                value = m(row, col);
                if (value < minValue)
                    minValue = value;
            }

            /* subtract the smallest element from each element of the column */
            for (row = 0; row<nOfRows; row++)
                m(row, col) -= minValue;
        }

        /* Steps 1 and 2a */
        for (col = 0; col<nOfColumns; col++)
            for (row = 0; row<nOfRows; row++)
                if (m(row, col) == 0)
                    if (!coveredRows(row))
                    {
                        starMatrix(row, col) = true;
                        coveredColumns(col) = true;
                        coveredRows(row) = true;
                        break;
                    }
        coveredRows.fill(0);

    }

    /* move to step 2b */
    step2b(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    return;
}

void step2a(VectorXi& assignment, MatrixXd& m, MatrixXb& starMatrix, MatrixXb& newStarMatrix, MatrixXb& primeMatrix, 
            VectorXi& coveredColumns, VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    /* cover every column containing a starred zero */
    for (int col = 0; col<nOfColumns; col++)
    {
        for (int row = 0; row < nOfRows; row++)
        {
            if (starMatrix(row, col))
            {
                coveredColumns(col) = true;
                break;
            }
        }
    }

    /* move to step 3 */
    step2b(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void step2b(VectorXi& assignment, MatrixXd& m, MatrixXb& starMatrix, MatrixXb& newStarMatrix, MatrixXb& primeMatrix, 
            VectorXi& coveredColumns, VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    /* count covered columns */
    int nOfCoveredColumns = 0;
    for (int col = 0; col<nOfColumns; col++)
        if (coveredColumns(col))
            nOfCoveredColumns++;

    if (nOfCoveredColumns == minDim)
    {
        /* algorithm finished */
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else
    {
        /* move to step 3 */
        step3(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

}

void step3(VectorXi& assignment, MatrixXd& m, MatrixXb& starMatrix, MatrixXb& newStarMatrix, MatrixXb& primeMatrix, 
           VectorXi& coveredColumns, VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    int starCol;
    bool zerosFound = true;
    while (zerosFound)
    {
        zerosFound = false;
        for (int col = 0; col<nOfColumns; col++)
            if (!coveredColumns(col))
                for (int row = 0; row<nOfRows; row++)
                    if ((!coveredRows(row)) && (m(row, col) == 0))
                    {
                        /* prime zero */
                        primeMatrix(row, col) = true;

                        /* find starred zero in current row */
                        for (starCol = 0; starCol<nOfColumns; starCol++)
                            if (starMatrix(row, starCol))
                                break;

                        if (starCol == nOfColumns) /* no starred zero found */
                        {
                            /* move to step 4 */
                            step4(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else
                        {
                            coveredRows(row) = true;
                            coveredColumns(starCol) = false;
                            zerosFound = true;
                            break;
                        }
                    }
    }

    /* move to step 5 */
    step5(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void step4(VectorXi& assignment, MatrixXd& m, MatrixXb& starMatrix, MatrixXb& newStarMatrix, MatrixXb& primeMatrix,
           VectorXi& coveredColumns, VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
    int starRow, starCol, primeRow, primeCol;
    int nOfElements = nOfRows*nOfColumns;

    /* generate temporary copy of starMatrix */
    for (int n = 0; n<nOfElements; n++)
        newStarMatrix(n) = starMatrix(n);

    /* star current zero */
    newStarMatrix(row, col) = true;

    /* find starred zero in current column */
    starCol = col;
    for (starRow = 0; starRow<nOfRows; starRow++)
        if (starMatrix(starRow, starCol))
            break;

    while (starRow<nOfRows)
    {
        /* unstar the starred zero */
        newStarMatrix(starRow, starCol) = false;

        /* find primed zero in current row */
        primeRow = starRow;
        for (primeCol = 0; primeCol<nOfColumns; primeCol++)
            if (primeMatrix(primeRow, primeCol))
                break;

        /* star the primed zero */
        newStarMatrix(primeRow, primeCol) = true;

        /* find starred zero in current column */
        starCol = primeCol;
        for (starRow = 0; starRow<nOfRows; starRow++)
            if (starMatrix(starRow, starCol))
                break;
    }

    /* use temporary copy as new starMatrix */
    /* delete all primes, uncover all rows */
    for (int n = 0; n<nOfElements; n++)
    {
        primeMatrix(n) = false;
        starMatrix(n) = newStarMatrix(n);
    }
    for (int n = 0; n<nOfRows; n++)
        coveredRows(n) = false;

    /* move to step 2a */
    step2a(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
 
void step5(VectorXi& assignment, MatrixXd& m, MatrixXb& starMatrix, MatrixXb& newStarMatrix, MatrixXb& primeMatrix, 
           VectorXi& coveredColumns, VectorXi& coveredRows, int nOfRows, int nOfColumns, int minDim)
{
    double h, value;
    int row, col;

    /* find smallest uncovered element h */
    h = std::numeric_limits<int>::max();
    for (row = 0; row<nOfRows; row++)
        if (!coveredRows(row))
            for (col = 0; col<nOfColumns; col++)
                if (!coveredColumns(col))
                {
                    value = m(row, col);
                    if (value < h)
                        h = value;
                }

    /* add h to each covered row */
    for (row = 0; row<nOfRows; row++)
        if (coveredRows(row))
            for (col = 0; col<nOfColumns; col++)
                m(row, col) += h;

    /* subtract h from each uncovered column */
    for (col = 0; col<nOfColumns; col++)
        if (!coveredColumns(col))
            for (row = 0; row<nOfRows; row++)
                m(row, col) -= h;

    /* move to step 3 */
    step3(assignment, m, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void buildassignmentvector(VectorXi& assignment, MatrixXb& starMatrix, int nOfRows, int nOfColumns)
{
    for (int row = 0; row<nOfRows; row++)
        for (int col = 0; col<nOfColumns; col++)
            if (starMatrix(row, col))
            {
                assignment(row) = col;
                break;
            }
}
