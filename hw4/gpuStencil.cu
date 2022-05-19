#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    
    int border = order / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < nx*ny) {
	int x = border + (i % nx);
	int y = border + (i / nx);
        int idx = gx * y + x;   
        next[idx] = Stencil<order>(&curr[idx], gx, xcfl, ycfl);
    
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    // initialize grid and bc
    boundary_conditions BC(params);
    Grid next_grid(curr_grid);

    // declare compute parameters
    int nx = params.nx();
    int ny = params.ny();
    int gx = params.gx();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    // declare block and grid dims
    const int block_size = 1024;
    const int grid_size = (nx*ny + block_size - 1) / block_size;
    
    // benchmark timing
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);
        
	// apply stencil
        if (params.order() == 2) {
            gpuStencilGlobal<2><<<grid_size, block_size>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        } 
	
	else if (params.order() == 4) {
            gpuStencilGlobal<4><<<grid_size, block_size>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        } 
	
	else if (params.order() == 8) {
            gpuStencilGlobal<8><<<grid_size, block_size>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }
 
        // update current grid	
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");
    double timing = stop_timer(&timer);

    // save results
    curr_grid.fromGPU();
    curr_grid.saveStateToFile("global_out.csv");

    return timing;

}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    
    
    int border = (int) (order / 2);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (blockIdx.y * blockDim.y + threadIdx.y)*numYPerStep;

    if (i < nx) {
	int x = i + border;  // x coordinate of matrix
        int niter = min(numYPerStep, ny-j);  // number of updates thread computes
	for (int it = 0; it < niter; it++) {
	    int y = j + it + border;
            int idx = gx * y + x;
            next[idx] = Stencil<order>(&curr[idx], gx, xcfl, ycfl);
	}
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);
    Grid next_grid(curr_grid);
    
    // load compute parameters
    int nx = params.nx();
    int ny = params.ny();
    int gx = params.gx();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    // declare block dim
    #define numYPerStep 2
    int nthreads = 256;
    int xthreads = 64;
    int ythreads = nthreads/xthreads;
    dim3 threads(xthreads, ythreads);

    // declare grid dim
    int xblocks = (nx + threads.x - 1)/threads.x; 
    int yblocks = (ny + threads.y - 1)/threads.y; 
    dim3 blocks(xblocks, yblocks);
    
    // benchmark timing
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        
	// update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);
        
	// apply stencil
        if (params.order() == 2) {
            gpuStencilBlock<2, numYPerStep><<<blocks, threads>>>(
	        next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }

	else if (params.order() == 4) {
            gpuStencilBlock<4, numYPerStep><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
        }

	else if (params.order() == 8) {
            gpuStencilBlock<8, numYPerStep><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl );
        }
        
	// update current grid
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    
    // map thread to global position
    int border = order / 2;
    int numY = side / blockDim.y;
    int sub_square_side = side - order;
    int i = blockIdx.x*sub_square_side + threadIdx.x;
    int j = blockIdx.y*sub_square_side + threadIdx.y*numY;
    
    // load mesh grid into shared memory
    __shared__ float shared[side][side];
    if (i < gx) 
    {
        int niter = min(numY, gy-j);  // number of updates thread computes 
	for (int it = 0; it < niter; it++) {
	    if ((j+it) < gy) {
	        shared[threadIdx.y*numY+it][threadIdx.x] = curr[gx*(j+it)+i];
	    }
	}
    }
    __syncthreads(); 

    // apply stencil inside domain
    if (i < (gx-border) &&
	threadIdx.x >= border && 
	threadIdx.x < (side-border)) 
    {
        int niter = min(numY, gy-j);  // number of updates thread computes 
	for (int it = 0; it < niter; it++) {
	    if ((j+it) < (gy-border) &&
	        (threadIdx.y*numY+it) >= border &&
                (threadIdx.y*numY+it) < (side-border)) 
	    {
	        next[gx*(j+it)+i] = Stencil<order>(
	            &shared[threadIdx.y*numY+it][threadIdx.x], 
	            side, 
	            xcfl, 
	            ycfl);
	    }
	}
    }

}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);
    Grid next_grid(curr_grid);

    // load compute parameters
    int gx = params.gx();
    int gy = params.gy();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    // declare block dim
    #define side 64
    int numYPerThread = 4;
    dim3 threads(side, side/numYPerThread);

    // declare grid dim
    int sub_square = side - order;
    int xblocks = (gx + sub_square - 1) / sub_square;
    int yblocks = (gy + sub_square - 1) / sub_square;
    dim3 blocks(xblocks, yblocks);
    
    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
    
	// update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // apply stencil
        if (order == 2) {
            gpuStencilShared<side, 2><<<blocks, threads>>>(
	        next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        }

	else if (order == 4) {
            gpuStencilShared<side, 4><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
        }

	else if (order == 8) {
            gpuStencilShared<side, 8><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl );
        }
        
	// update current grid
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}

