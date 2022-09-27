#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <algorithm>
#include <memory>
#include <limits>

#include "Utils/SpUtils.hpp"
#include "Legacy/SpRuntime.hpp"


#include "CholeskyFunctionsWrapper.hpp"

struct Block{
    int rowOffset;
    int colOffset;

    int nbRows;
    int nbCols;

    std::unique_ptr<double[]> values;
    std::unique_ptr<int[]> permutations;
};

//////////////////////////////////////////////////////////////////////////////

std::unique_ptr<double[]> generatePositiveDefinitMatrix(const int inMatrixDim){
    std::unique_ptr<double[]> matrix(new double[inMatrixDim*inMatrixDim]());

    srand48(0);

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            matrix[idxRow*inMatrixDim+idxCol] = drand48();
            matrix[idxCol*inMatrixDim+idxRow] = matrix[idxRow*inMatrixDim+idxCol];
        }
    }

    for(int idxDiag = 0 ; idxDiag < inMatrixDim ; ++idxDiag){
        matrix[idxDiag*inMatrixDim+idxDiag] += inMatrixDim;
    }

    return matrix;
}

std::unique_ptr<double[]> generateMatrixLikeStarpu(const int inMatrixDim){
    std::unique_ptr<double[]> matrix(new double[inMatrixDim*inMatrixDim]());

    srand48(0);

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            matrix[idxCol*inMatrixDim+idxRow] = (1.0/double(1+idxRow+idxCol)) + ((idxRow == idxCol)?double(inMatrixDim):0.0);
        }
    }

    return matrix;
}

std::unique_ptr<Block[]> matrixToBlock(double matrix[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    std::unique_ptr<Block[]> blocks(new Block[nbBlocks*nbBlocks]);

    // Init the blocks
    for(int m = 0 ; m < nbBlocks ; ++m){
        for(int n = 0 ; n < nbBlocks ; ++n){
            blocks[m*nbBlocks+n].rowOffset = m*inBlockDim;
            blocks[m*nbBlocks+n].colOffset = n*inBlockDim;
            blocks[m*nbBlocks+n].nbRows = std::min(inMatrixDim - m*inBlockDim, inBlockDim);
            blocks[m*nbBlocks+n].nbCols = std::min(inMatrixDim - n*inBlockDim, inBlockDim);
            blocks[m*nbBlocks+n].values.reset(new double[blocks[m*nbBlocks+n].nbRows * blocks[m*nbBlocks+n].nbCols]());
            blocks[m*nbBlocks+n].permutations.reset(new int[blocks[m*nbBlocks+n].nbRows]());
        }
    }

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            const int colIdxInBlock = idxCol%inBlockDim;

            blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock] = matrix[idxRow*inMatrixDim+idxCol];
        }
    }

    return blocks;
}

double diffMatrixBlocks(double matrix[], Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    double error = 0;

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            const int colIdxInBlock = idxCol%inBlockDim;

            const double blockValue = blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock];
            const double matrixValue = matrix[idxRow*inMatrixDim+idxCol];

            error = std::max(error, std::abs(blockValue-matrixValue)/(std::abs(matrixValue)+std::numeric_limits<double>::epsilon()));
        }
    }

    return error;
}

void printMatrix(const double matrix[], const int inMatrixDim){
    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        std::cout << idxRow << "]\t";
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            std::cout << matrix[idxRow*inMatrixDim+idxCol] << "\t";
        }
        std::cout << "\n";
    }
}

void printBlocks(const Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;
    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        std::cout << idxRow << "]\t";
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            const int colIdxInBlock = idxCol%inBlockDim;

            const double blockValue = blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock];
            std::cout << blockValue << "\t";
        }
        std::cout << "\n";
    }
}


//////////////////////////////////////////////////////////////////////////////

void choleskyFactorizationMatrix(double matrix[], const int inMatrixDim){
    Cholesky::potrf( Cholesky::FillMode::LWPR, inMatrixDim, matrix, inMatrixDim );
}

void choleskyFactorization(Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    // Compute the blocks
    for(int k = 0 ; k < nbBlocks ; ++k){
        // TODO put for syrk and gemm ? const double rbeta = (j==0) ? beta : 1.0;
        // POTRF( RW A(k,k) )
        runtime.task(SpPriority(0), SpWrite(blocks[k*nbBlocks+k]),
                [inBlockDim](Block& block){
            Cholesky::potrf( Cholesky::FillMode::LWPR, inBlockDim, block.values.get(), inBlockDim );
        }).setTaskName(std::string("potrf -- (W-")+std::to_string(k)+","+std::to_string(k)+")");

        for(int m = k + 1 ; m < nbBlocks ; ++m){
            // TRSM( R A(k,k), RW A(m, k) )
            runtime.task(SpPriority(1), SpRead(blocks[k*nbBlocks+k]), SpWrite(blocks[k*nbBlocks+m]),
                    [inBlockDim](const Block& blockA, Block& blockB){
                Cholesky::trsm( Cholesky::Side::RIGHT, Cholesky::FillMode::LWPR,
                                Cholesky::Transa::TRANSPOSE, Cholesky::DiagUnit::NON_UNIT_TRIANGULAR,
                                inBlockDim, inBlockDim, 1.0, blockA.values.get(), inBlockDim,
                                blockB.values.get(), inBlockDim );
            }).setTaskName(std::string("TRSM -- (R-")+std::to_string(k)+","+std::to_string(k)+") (W-"+std::to_string(m)+","+std::to_string(k)+")");
        }

        for(int n = k+1 ; n < nbBlocks ; ++n){
            // SYRK( R A(n,k), RW A(n, n) )
            runtime.task(SpPriority(1), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[n*nbBlocks+n]),
                    [inBlockDim](const Block& blockA, Block& blockC){
                Cholesky::syrk( Cholesky::FillMode::LWPR,
                                Cholesky::Transa::NORMAL,
                                inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                1.0, blockC.values.get(), inBlockDim );
            }).setTaskName(std::string("SYRK -- (R-")+std::to_string(n)+","+std::to_string(k)+") (W-"+std::to_string(n)+","+std::to_string(n)+")");

            for(int m = k+1 ; m < nbBlocks ; ++m){
                // GEMM( R A(m, k), R A(n, k), RW A(m, n))
                runtime.task(SpPriority(3), SpRead(blocks[k*nbBlocks+m]), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[m*nbBlocks+n]),
                        [inBlockDim](const Block& blockA, const Block& blockB, Block& blockC){
                    Cholesky::gemm( Cholesky::Transa::NORMAL, Cholesky::Transa::TRANSPOSE,
                                    inBlockDim, inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                    blockB.values.get(), inBlockDim,
                                    1.0, blockC.values.get(), inBlockDim );
                }).setTaskName(std::string("GEMM -- (R-")+std::to_string(m)+","+std::to_string(k)+")(R-"+std::to_string(n)+","+std::to_string(k)+")(W-"+std::to_string(m)+","+std::to_string(n)+")");
            }
        }
    }

    runtime.waitAllTasks();
    runtime.stopAllThreads();
    runtime.generateDot("/tmp/graph.dot");
}

int main(){
    const int MatrixSize = 16;
    const int BlockSize = 2;
    const bool printValues = (MatrixSize <= 16);
    /////////////////////////////////////////////////////////
    auto matrix = generateMatrixLikeStarpu(MatrixSize);// generatePositiveDefinitMatrix(MatrixSize);
    if(printValues){
        std::cout << "Matrix:\n";
        printMatrix(matrix.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    auto blocks = matrixToBlock(matrix.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks:\n";
        printBlocks(blocks.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterCopy = diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after copy : " << errorAfterCopy << std::endl;
    /////////////////////////////////////////////////////////
    choleskyFactorization(blocks.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks after facto:\n";
        printBlocks(blocks.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    choleskyFactorizationMatrix(matrix.get(), MatrixSize);
    if(printValues){
        std::cout << "Matrix after facto:\n";
        printMatrix(matrix.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterFacto = diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;

    return 0;
}
