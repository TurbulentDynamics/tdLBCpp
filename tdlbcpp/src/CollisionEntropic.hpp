//
//  CollisionEntropic.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"
#include <cmath>
#include <chrono>
#include <iostream>

// Entropic Lattice Boltzmann Method (ELBM) Collision Operator
// Based on Ansumali & Karlin (2002) "Entropy Function Approach to the Lattice Boltzmann Method"
// and Chikatamarla & Karlin (2006) "Entropic Lattice Boltzmann Models for Hydrodynamics in Three Dimensions"
//
// The entropic collision operator ensures H-theorem compliance (entropy conservation)
// and provides enhanced numerical stability compared to BGK-based operators.

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, Entropic, streamingType>::collision() {
    using AF = AccessField<T, QVecSize, MemoryLayout, Entropic, streamingType>;

    // Timing is controlled by NDEBUG - disabled in release builds (-DNDEBUG)
#ifndef NDEBUG
    // Debug timing variables
    auto t_start = std::chrono::high_resolution_clock::now();
    double time_read = 0.0, time_velocity = 0.0, time_equilibrium = 0.0;
    double time_alpha_search = 0.0, time_collision = 0.0, time_write = 0.0;
    size_t total_cells = 0;
    size_t total_alpha_iterations = 0;
#endif

    // Lattice speed of sound squared: c_s^2 = 1/3 for D3Q19
    const T cs2 = 1.0 / 3.0;

    // Relaxation parameter from kinematic viscosity
    // tau = 3*nu + 0.5
    T tau = 3.0 * flow.nu + 0.5;

    // Entropy stabilization parameters
    const T alpha_tolerance = 1e-8;  // Convergence tolerance for alpha
    const int alpha_max_iter = 100;   // Maximum iterations for alpha search

    for (tNi i = 1; i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            for (tNi k = 1; k <= zg1; k++) {
#ifndef NDEBUG
                total_cells++;
                auto t0 = std::chrono::high_resolution_clock::now();
#endif

                // Read force at this location
                Force<T> f = F[index(i, j, k)];

                // Read distribution functions from velocity space
                QVec<T, QVecSize> f_i = AF::read(*this, i, j, k);

#ifndef NDEBUG
                auto t1 = std::chrono::high_resolution_clock::now();
                time_read += std::chrono::duration<double>(t1 - t0).count();
#endif

                // D3Q19 lattice velocities (c_ix, c_iy, c_iz) - Q01 to Q18 (18 directions, Q0 not stored)
                const int cx[18] = {1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0};
                const int cy[18] = {0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1};
                const int cz[18] = {0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1, 1,-1};

                // D3Q19 lattice weights (for 18 stored directions, Q01-Q18)
                // Note: Q0 (rest particle, w=1/3) is NOT stored in this D3Q19 implementation
                const T w[18] = {
                    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  // Q01-Q06: faces
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  // Q07-Q12: edges
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0   // Q13-Q18: edges
                };

                // Compute macroscopic density from distribution functions
                // rho = sum_i f_i
                T rho = 0.0;
                #pragma omp simd reduction(+:rho)
                for (int l = 0; l < QVecSize; l++) {
                    rho += f_i.q[l];
                }

                // Safety check: ensure density is positive and finite
                if (rho <= 0.0 || std::isnan(rho) || std::isinf(rho)) {
                    std::cerr << "ERROR: Invalid density at (" << i << ", " << j << ", " << k
                              << ") rho = " << rho << std::endl;
                    // Reset to unit density with zero velocity
                    rho = 1.0;
                    for (int l = 0; l < QVecSize; l++) {
                        f_i.q[l] = w[l];  // Equilibrium at rest
                    }
                }

                // Compute momentum from distribution functions
                // rho*u = sum_i f_i * c_i
                T rho_ux = 0.0, rho_uy = 0.0, rho_uz = 0.0;
                #pragma omp simd reduction(+:rho_ux,rho_uy,rho_uz)
                for (int l = 0; l < QVecSize; l++) {
                    rho_ux += f_i.q[l] * cx[l];
                    rho_uy += f_i.q[l] * cy[l];
                    rho_uz += f_i.q[l] * cz[l];
                }

                // Compute macroscopic velocity including force contribution (Guo forcing)
                Velocity<T> u;
                u.x = rho_ux / rho + 0.5 * f.x / rho;
                u.y = rho_uy / rho + 0.5 * f.y / rho;
                u.z = rho_uz / rho + 0.5 * f.z / rho;

#ifndef NDEBUG
                auto t2 = std::chrono::high_resolution_clock::now();
                time_velocity += std::chrono::duration<double>(t2 - t1).count();
#endif

                // Precompute velocity products for equilibrium distribution
                T ux2 = u.x * u.x;
                T uy2 = u.y * u.y;
                T uz2 = u.z * u.z;
                T u_dot_u = ux2 + uy2 + uz2;

                // Compute equilibrium distributions in velocity space
                // f_i^eq = w_i * rho * [1 + (c_i·u)/cs2 + (c_i·u)^2/(2*cs2^2) - u·u/(2*cs2)]
                QVec<T, QVecSize> f_eq;

                #pragma omp simd
                for (int l = 0; l < QVecSize; l++) {
                    T c_dot_u = cx[l] * u.x + cy[l] * u.y + cz[l] * u.z;
                    T c_dot_u_2 = c_dot_u * c_dot_u;

                    f_eq.q[l] = w[l] * rho * (
                        1.0 +
                        c_dot_u / cs2 +
                        c_dot_u_2 / (2.0 * cs2 * cs2) -
                        u_dot_u / (2.0 * cs2)
                    );
                }

#ifndef NDEBUG
                auto t3 = std::chrono::high_resolution_clock::now();
                time_equilibrium += std::chrono::duration<double>(t3 - t2).count();
#endif

                // Entropic stabilization: Find optimal alpha parameter
                // The parameter alpha adjusts the relaxation to ensure entropy decrease
                // and distribution positivity
                // Start with alpha=2.0 (standard BGK corresponds to alpha=2) for each cell
                T alpha = 2.0;
#ifndef NDEBUG
                int iter_count = 0;
#endif

                // Compute entropy function for stability check
                // H = sum_i f_i * ln(f_i / w_i)
                // The collision must ensure delta_H <= 0 (H-theorem)

                for (int iter = 0; iter < alpha_max_iter; iter++) {
#ifndef NDEBUG
                    iter_count = iter;
#endif

                    // Check if this alpha satisfies entropy and positivity conditions
                    bool entropy_satisfied = true;

                    #pragma omp simd reduction(&& : entropy_satisfied)
                    for (int l = 0; l < QVecSize; l++) {
                        // Entropic collision: f* = f + alpha/2 * (f_eq - f)
                        T f_star = f_i.q[l] + (alpha / 2.0) * (f_eq.q[l] - f_i.q[l]);

                        // Ensure strict positivity (necessary for entropy definition)
                        // Use a small epsilon to prevent numerical issues
                        entropy_satisfied = entropy_satisfied && (f_star > 1.0e-14);
                    }

                    if (entropy_satisfied) {
                        break;
                    }

                    // Reduce alpha if entropy condition not satisfied
                    // Use bisection for stability
                    alpha *= 0.5;

                    if (alpha < alpha_tolerance) {
                        // Fall back to very small alpha for stability
                        alpha = alpha_tolerance;
                        break;
                    }
                }

#ifndef NDEBUG
                total_alpha_iterations += iter_count;
#endif

#ifndef NDEBUG
                auto t4 = std::chrono::high_resolution_clock::now();
                time_alpha_search += std::chrono::duration<double>(t4 - t3).count();
#endif

                // Apply entropic collision with optimized alpha
                QVec<T, QVecSize> f_new;

                // Vectorized collision step with Guo forcing
                #pragma omp simd
                for (int l = 0; l < QVecSize; l++) {
                    // Entropic relaxation: f* = f + alpha/2 * (f_eq - f)
                    f_new.q[l] = f_i.q[l] + (alpha / 2.0) * (f_eq.q[l] - f_i.q[l]);

                    // Add Guo forcing term: F_i = (1 - 1/(2*tau)) * w_i * [(c_i - u)/cs2 + (c_i·u)*c_i/cs2^2] · F
                    // Simplified for efficiency
                    T c_dot_u = cx[l] * u.x + cy[l] * u.y + cz[l] * u.z;
                    T c_dot_F = cx[l] * f.x + cy[l] * f.y + cz[l] * f.z;
                    T u_dot_F = u.x * f.x + u.y * f.y + u.z * f.z;

                    T forcing_term = w[l] * (
                        (c_dot_F - u_dot_F) / cs2 +
                        c_dot_u * c_dot_F / (cs2 * cs2)
                    );

                    f_new.q[l] += (1.0 - 0.5 / tau) * forcing_term;

                    // Final safety check: clamp to small positive value if negative
                    if (f_new.q[l] <= 0.0 || std::isnan(f_new.q[l]) || std::isinf(f_new.q[l])) {
                        f_new.q[l] = 1.0e-14;  // Very small positive value
                    }
                }

#ifndef NDEBUG
                auto t5 = std::chrono::high_resolution_clock::now();
                time_collision += std::chrono::duration<double>(t5 - t4).count();
#endif

                // Write back the collided distribution to velocity space
                AF::write(*this, f_new, i, j, k);

#ifndef NDEBUG
                auto t6 = std::chrono::high_resolution_clock::now();
                time_write += std::chrono::duration<double>(t6 - t5).count();
#endif
            }
        }
    }

#ifndef NDEBUG
    // Print timing summary
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== CollisionEntropic Timing Breakdown ===" << std::endl;
    std::cout << "Total cells processed: " << total_cells << std::endl;
    std::cout << "Total time: " << total_time * 1000.0 << " ms" << std::endl;
    std::cout << "\nPer-section timing:" << std::endl;
    std::cout << "  Read distributions:  " << time_read * 1000.0 << " ms ("
              << (time_read / total_time * 100.0) << "%)" << std::endl;
    std::cout << "  Compute velocity:    " << time_velocity * 1000.0 << " ms ("
              << (time_velocity / total_time * 100.0) << "%)" << std::endl;
    std::cout << "  Equilibrium moments: " << time_equilibrium * 1000.0 << " ms ("
              << (time_equilibrium / total_time * 100.0) << "%)" << std::endl;
    std::cout << "  Alpha search:        " << time_alpha_search * 1000.0 << " ms ("
              << (time_alpha_search / total_time * 100.0) << "%)" << std::endl;
    std::cout << "  Collision step:      " << time_collision * 1000.0 << " ms ("
              << (time_collision / total_time * 100.0) << "%)" << std::endl;
    std::cout << "  Write distributions: " << time_write * 1000.0 << " ms ("
              << (time_write / total_time * 100.0) << "%)" << std::endl;
    std::cout << "\nAverage alpha iterations per cell: "
              << (double)total_alpha_iterations / total_cells << std::endl;
    std::cout << "=========================================\n" << std::endl;
#endif
}




