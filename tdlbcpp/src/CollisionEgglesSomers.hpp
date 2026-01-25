//
//  CollisionEgglesSomers.hpp
//  tdLB
//
//  Created by Niall Ó Broin on 08/12/2020.
//

#pragma once

#include "ComputeUnit.h"

// Eggels & Somers Collision Operator for D3Q19 Lattice Boltzmann Method
//
// This file implements two variants of the Eggels & Somers collision operator:
//
// 1. EgglesSomersLES - WITH Large Eddy Simulation (LES) turbulence modeling
//    - Uses Smagorinsky subgrid-scale model for turbulence
//    - Dynamically computes eddy viscosity based on resolved strain rate
//    - Suitable for turbulent flows where grid resolution cannot capture all scales
//    - Reference: Eggels & Somers (1995), Somers (1993)
//
// 2. EgglesSomers - WITHOUT LES (Direct Numerical Simulation or laminar flows)
//    - Uses constant kinematic viscosity (no turbulence model)
//    - Suitable for laminar flows or DNS where all scales are resolved
//
// Key Features:
// - Moment-based collision in D3Q19 velocity space
// - Filter matrix approach (Eggels & Somers, 1995, eq. 12)
// - 3rd order compensation with g3 parameter
// - Guo forcing scheme for external body forces
//
// The collision operates in two phases:
// 1. Relaxation: moments relax toward equilibrium with filter matrix
// 2. Inverse transform: convert relaxed moments back to velocity distribution




// ============================================================================
// EgglesSomersLES - Collision Operator WITH Large Eddy Simulation
// ============================================================================
//
// Implements the Eggels & Somers collision operator with Smagorinsky LES
// turbulence modeling for turbulent flows.
//
// LES Model:
//   - Smagorinsky subgrid-scale (SGS) model
//   - Eddy viscosity: nu_t = (C_s * delta)^2 * sqrt(2 * S_ij * S_ij)
//   - C_s (Smagorinsky constant) = flow.cs0 (typically 0.12)
//   - Total viscosity: nu_total = nu + nu_t
//   - Strain rate computed from resolved velocity gradients
//
// Key Difference from EgglesSomers:
//   - Dynamically computes eddy viscosity at each grid point
//   - Updates relaxation parameters b and c based on total viscosity
//   - Accounts for subgrid-scale turbulent energy dissipation
//
// Suitable for:
//   - High Reynolds number turbulent flows
//   - Flows where grid resolution insufficient for DNS
//   - Mixing vessels, channel flows, jets with turbulence

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomersLES, streamingType>::collision(){
    using AF = AccessField<T, QVecSize, MemoryLayout, EgglesSomersLES, streamingType>;

    // Initial relaxation parameters based on molecular viscosity only
    // These will be updated with eddy viscosity for each cell
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){


                Force<T> f = F[index(i,j,k)];


                QVec<T, QVecSize> m = AF::read(*this, i, j, k);


                Velocity<T> u = m.velocity(f);

                QVec<T, QVecSize> alpha;


                // ========================================================
                // SMAGORINSKY LES TURBULENCE MODEL
                // ========================================================
                //
                // Compute strain rate tensor from moments and calculate
                // eddy viscosity using Smagorinsky model.
                //
                // The strain rate is computed from the velocity gradients,
                // which are extracted from the non-equilibrium moments.
                // This follows Somers (1993) approach where low strain
                // rates do not excite eddy viscosity.

                // Scale factor for computing velocity gradients from moments
                T fct = 3.0 / (m.q[M01] * (1.0 + 6.0 * (Nu[index(i,j,k)] + flow.nu)));

                // Diagonal components of velocity gradient tensor
                // ∂u/∂x, ∂v/∂y, ∂w/∂z computed from non-equilibrium moments
                T dudx = fct * ((m.q[M02] + 0.5 * F[index(i,j,k)].x * u.x - m.q[M05]));
                T dvdy = fct * ((m.q[M03] + 0.5 * F[index(i,j,k)].y * u.y - m.q[M07]));
                T dwdz = fct * ((m.q[M04] + 0.5 * F[index(i,j,k)].z * u.z - m.q[M10]));

                // Divergence of velocity field (should be ~0 for incompressible)
                T divv = dudx + dvdy + dwdz;


                // Off-diagonal components of velocity gradient tensor
                // ∂u/∂y + ∂v/∂x, ∂u/∂z + ∂w/∂x, ∂v/∂z + ∂w/∂y
                T dudypdvdx = 2 * fct * ((m.q[M03]) + 0.5 * F[index(i,j,k)].y * u.x - m.q[M06]);
                T dudzpdwdx = 2 * fct * ((m.q[M04]) + 0.5 * F[index(i,j,k)].z * u.x - m.q[M08]);
                T dvdzpdwdy = 2 * fct * ((m.q[M04]) + 0.5 * F[index(i,j,k)].z * u.y - m.q[M09]);


                // Resolved strain rate magnitude: S^2 = 2 * S_ij * S_ij
                // where S_ij is the strain rate tensor = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
                // The trace-free formulation subtracts (2/3)*div(v)^2 for proper
                // treatment of compressibility effects
                T sh = 2 * dudx*dudx + 2 * dvdy*dvdy + 2 * dwdz*dwdz + dudypdvdx*dudypdvdx + dudzpdwdx*dudzpdwdx + dvdzpdwdy*dvdzpdwdy - (2.0 / 3.0) * divv*divv;


                // Smagorinsky eddy viscosity calculation:
                // nu_t = (C_s * Δ)^2 * |S|
                // where:
                //   C_s = flow.cs0 (Smagorinsky constant, typically 0.12)
                //   Δ = grid spacing (implicitly 1 in lattice units)
                //   |S| = sqrt(2 * S_ij * S_ij) = sqrt(sh)
                Nu[index(i,j,k)] = flow.cs0 * flow.cs0 * sqrt(fabs(sh));



                // Update total effective viscosity (molecular + turbulent)
                // and recompute relaxation parameters b and c
                //
                // Key feature of Somers (1993) approach:
                // - Low strain rates → small eddy viscosity → near-laminar behavior
                // - High strain rates → large eddy viscosity → enhanced dissipation
                //
                // This dynamic adjustment allows the model to adapt to local
                // flow conditions without requiring a priori knowledge of
                // turbulence intensity.

                T nut = Nu[index(i,j,k)] + flow.nu;  // Total kinematic viscosity
                b = 1.0 / (1.0 + 6 * nut);            // Updated relaxation parameter
                c = 1.0 - 6 * nut;                    // Updated complement parameter


                // ========================================================
                // MOMENT RELAXATION (COLLISION STEP)
                // ========================================================
                //
                // Relax moments toward equilibrium using the filter matrix
                // approach from Eggels & Somers (1995).
                //
                // The relaxation process:
                // 1. 0th order (mass): conserved exactly (no relaxation)
                // 2. 1st order (momentum): includes external forcing (Guo scheme)
                // 3. 2nd order (stress): relaxed toward equilibrium with viscosity
                // 4. 3rd order: compensated with g3 parameter for stability
                // 5. 4th order and higher: set to zero (hydrodynamic limit)

                // 0th order moment: mass density (conserved)
                alpha[M01] = m[M01];


                // 1st order moments: momentum with external forcing
                // Forcing added using Guo scheme for improved Galilean invariance
                alpha[M02] = m[M02] + f.x;
                alpha[M03] = m[M03] + f.y;
                alpha[M04] = m[M04] + f.z;

                // 2nd order moments: stress tensor relaxation
                // Equilibrium form: m_eq = rho * u_i * u_j (off-diagonal)
                //                   m_eq = rho * (u_i^2 - c_s^2) (diagonal)
                // Relaxation: alpha = b * (2*m_eq - c*m)
                // where b and c depend on viscosity (molecular + turbulent for LES)
                alpha[M05] = (2.0 * (m[M02] + 0.5 * f.x) * u.x - m[M05]*c)*b;
                alpha[M06] = (2.0 * (m[M02] + 0.5 * f.x) * u.y - m[M06]*c)*b;
                alpha[M07] = (2.0 * (m[M03] + 0.5 * f.y) * u.y - m[M07]*c)*b;

                alpha[M08] = (2.0 * (m[M02] + 0.5 * f.x) * u.z - m[M08]*c)*b;
                alpha[M09] = (2.0 * (m[M03] + 0.5 * f.y) * u.z - m[M09]*c)*b;
                alpha[M10] = (2.0 * (m[M04] + 0.5 * f.z) * u.z - m[M10]*c)*b;

                // 3rd order moments: compensation with g3 parameter
                // The g3 parameter (typically 0.8) compensates for third-order
                // errors and improves numerical stability (Eggels & Somers 1995)
                #pragma omp simd
                for (int l = M11; l <= M16; l++) {
                    alpha[l] = -flow.g3 * m[l];
                }

                // 4th order moments: set to zero (hydrodynamic limit)
                alpha[M17] = 0.0;
                alpha[M18] = 0.0;


                // ========================================================
                // INVERSE MOMENT TRANSFORM (FILTER MATRIX APPLICATION)
                // ========================================================
                //
                // Transform relaxed moments (alpha) back to velocity space
                // distributions using the inverse filter matrix E.
                //
                // This implements Eq. 12 from Eggels & Somers (1995):
                //   f_i = Σ_j E_ij * alpha_j
                //
                // The filter matrix E is hardcoded in the coefficients below.
                // Each coefficient comes from the D3Q19 lattice symmetry and
                // orthogonality properties of the moment basis.

                // Scale factor for inverse transform (from D3Q19 normalization)
                #pragma omp simd
                for (int l=0;  l<QVecSize; l++) {
                    alpha[l] /= 24.0;
                }


                m[Q01] = 2*alpha[M01] + 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M11] - 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q02] = 2*alpha[M01] - 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M11] + 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q03] = 2*alpha[M01] + 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M12] - 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q04] = 2*alpha[M01] - 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M12] + 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q05] = 2*alpha[M01] + 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] - 4*alpha[M15] - 4*alpha[M17];

                m[Q06] = 2*alpha[M01] - 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] + 4*alpha[M15] - 4*alpha[M17];

                m[Q07] = alpha[M01] + 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M14] = alpha[M01] - 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M08] = alpha[M01] - 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M13] = alpha[M01] + 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M09] = alpha[M01] + 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M16] = alpha[M01] - 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M10] = alpha[M01] - 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M15] = alpha[M01] + 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M11] = alpha[M01] + 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M18] = alpha[M01] - 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M12] = alpha[M01] - 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];

                m[M17] = alpha[M01] + 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];




                AF::write(*this, m, i, j, k);

            }
        }
    }
}


// ============================================================================
// EgglesSomers - Collision Operator WITHOUT Large Eddy Simulation
// ============================================================================
//
// Implements the Eggels & Somers collision operator using only molecular
// (kinematic) viscosity, without any turbulence modeling.
//
// Viscosity Model:
//   - Constant kinematic viscosity: nu (from flow parameters)
//   - No eddy viscosity or subgrid-scale model
//   - Relaxation parameters b and c are constant throughout domain
//   - Direct solution of Navier-Stokes at the resolved scales
//
// Key Difference from EgglesSomersLES:
//   - Does NOT compute strain rate or eddy viscosity
//   - Uses constant relaxation parameters for all grid points
//   - Simpler and faster computation (no LES overhead)
//   - All dissipation from molecular viscosity only
//
// Suitable for:
//   - Laminar flows (low Reynolds number)
//   - Direct Numerical Simulation (DNS) with fine grid resolution
//   - Flows where turbulence can be neglected
//   - Cases where grid resolution sufficient to capture all scales
//
// NOT suitable for:
//   - Under-resolved turbulent flows (use EgglesSomersLES instead)
//   - High Reynolds number flows without adequate grid resolution

template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Streaming streamingType>
void ComputeUnitCollision<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>::collision(){
    using AF = AccessField<T, QVecSize, MemoryLayout, EgglesSomers, streamingType>;

    // Constant relaxation parameters based on molecular viscosity
    // These remain fixed throughout the simulation (no LES adjustment)
    T b = 1.0 / (1.0 + 6 * flow.nu);
    T c = 1.0 - 6 * flow.nu;


    for (tNi i=1; i<=xg1; i++){
        for (tNi j=1; j<=yg1; j++){
            for (tNi k=1; k<=zg1; k++){


                Force<T> f = F[index(i,j,k)];


                QVec<T, QVecSize> m = AF::read(*this, i, j, k);


                Velocity<T> u = m.velocity(f);

                QVec<T, QVecSize> alpha;



                //0th order term
                alpha[M01] = m[M01];


                //1st order term
                alpha[M02] = m[M02] + f.x;
                alpha[M03] = m[M03] + f.y;
                alpha[M04] = m[M04] + f.z;

                //2nd order terms
                alpha[M05] = (2.0 * (m[M02] + 0.5 * f.x) * u.x - m[M05]*c)*b;
                alpha[M06] = (2.0 * (m[M02] + 0.5 * f.x) * u.y - m[M06]*c)*b;
                alpha[M07] = (2.0 * (m[M03] + 0.5 * f.y) * u.y - m[M07]*c)*b;

                alpha[M08] = (2.0 * (m[M02] + 0.5 * f.x) * u.z - m[M08]*c)*b;
                alpha[M09] = (2.0 * (m[M03] + 0.5 * f.y) * u.z - m[M09]*c)*b;
                alpha[M10] = (2.0 * (m[M04] + 0.5 * f.z) * u.z - m[M10]*c)*b;

                //3rd order terms
                #pragma omp simd
                for (int l = M11; l <= M16; l++) {
                    alpha[l] = -flow.g3 * m[l];
                }

                //4th order terms
                alpha[M17] = 0.0;
                alpha[M18] = 0.0;


                // Start of invMoments, which is responsible for determining
                // the N-field (x) from alpha+ (alpha). It does this by using eq.
                // 12 in the article by Eggels and Somers (1995), which means
                // it's using the "filter matrix E" (not really present in the
                // code as matrix, but it's where the coefficients come from).

                #pragma omp simd
                for (int l=0;  l<QVecSize; l++) {
                    alpha[l] /= 24.0;
                }


                m[Q01] = 2*alpha[M01] + 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M11] - 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q02] = 2*alpha[M01] - 4*alpha[M02] + 3*alpha[M05] - 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M11] + 2*alpha[M13] + 2*alpha[M17] + 2*alpha[M18];

                m[Q03] = 2*alpha[M01] + 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] - 2*alpha[M12] - 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q04] = 2*alpha[M01] - 4*alpha[M03] - 3*alpha[M05] + 3*alpha[M07] - 3*alpha[M10] + 2*alpha[M12] + 2*alpha[M14] + 2*alpha[M17] - 2*alpha[M18];

                m[Q05] = 2*alpha[M01] + 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] - 4*alpha[M15] - 4*alpha[M17];

                m[Q06] = 2*alpha[M01] - 4*alpha[M04] - 3*alpha[M05] - 3*alpha[M07] + 3*alpha[M10] + 4*alpha[M15] - 4*alpha[M17];

                m[Q07] = alpha[M01] + 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M14] = alpha[M01] - 2*alpha[M02] + 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] + 2*alpha[M12] - 2*alpha[M17];

                m[M08] = alpha[M01] - 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] + 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] - 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M13] = alpha[M01] + 2*alpha[M02] - 2*alpha[M03] + 1.5*alpha[M05] - 6*alpha[M06] + 1.5*alpha[M07] - 1.5*alpha[M10] + 2*alpha[M11] - 2*alpha[M12] - 2*alpha[M17];

                m[M09] = alpha[M01] + 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M16] = alpha[M01] - 2*alpha[M02] + 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] + alpha[M15] - alpha[M16] + alpha[M17] - alpha[M18];

                m[M10] = alpha[M01] - 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] + 6*alpha[M08] + 1.5*alpha[M10] + alpha[M11] - alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M15] = alpha[M01] + 2*alpha[M02] - 2*alpha[M04] + 1.5*alpha[M05] - 1.5*alpha[M07] - 6*alpha[M08] + 1.5*alpha[M10] - alpha[M11] + alpha[M13] - alpha[M15] + alpha[M16] + alpha[M17] - alpha[M18];

                m[M11] = alpha[M01] + 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M18] = alpha[M01] - 2*alpha[M03] + 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] + alpha[M15] + alpha[M16] + alpha[M17] + alpha[M18];

                m[M12] = alpha[M01] - 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] + 6*alpha[M09] + 1.5*alpha[M10] + alpha[M12] - alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];

                m[M17] = alpha[M01] + 2*alpha[M03] - 2*alpha[M04] - 1.5*alpha[M05] + 1.5*alpha[M07] - 6*alpha[M09] + 1.5*alpha[M10] - alpha[M12] + alpha[M14] - alpha[M15] - alpha[M16] + alpha[M17] + alpha[M18];




                AF::write(*this, m, i, j, k);

            }
        }
    }
}



template <typename T, int QVecSize, MemoryLayoutType MemoryLayout, Collision collisionType, Streaming streamingType>
void ComputeUnitForcing<T, QVecSize, MemoryLayout, collisionType, streamingType>::moments(){

    using QVecAcc = QVecAccess<T, QVecSize, MemoryLayout>;
    using AF = AccessField<T, QVecSize, MemoryLayout, collisionType, streamingType>;

    for (tNi i = 1;  i <= xg1; i++) {
        for (tNi j = 1; j <= yg1; j++) {
            for (tNi k = 1; k <= zg1; k++) {


                QVec<T, QVecSize> q = AF::read(*this, i, j, k);


                QVec<T, QVecSize> m;

                //the first position is simply the entire mass-vector (Q summed up)
                m[M01] = q.q[Q01] + q.q[Q03] + q.q[Q02] + q.q[Q04] + q.q[Q05] + q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];


                //the second position is everything with an x-component
                m[M02] = q.q[Q01] - q.q[Q02] + q.q[Q07] - q.q[Q14] - q.q[Q08] + q.q[Q13] + q.q[Q09] - q.q[Q16] - q.q[Q10] + q.q[Q15];


                //the third position is everything with an y-component
                m[M03] = q.q[Q03] - q.q[Q04] + q.q[Q07] + q.q[Q14] - q.q[Q08] - q.q[Q13] + q.q[Q11] - q.q[Q18] - q.q[Q12] + q.q[Q17];


                //the fourth position is everything with a z-component
                m[M04] = q.q[Q05] - q.q[Q06] + q.q[Q09] + q.q[Q16] - q.q[Q10] - q.q[Q15] + q.q[Q11] + q.q[Q18] - q.q[Q12] - q.q[Q17];


                //starting from the fifth position, it gets more complicated in
                //structure, but it still follows the article by Eggels and Somers
                m[M05] =  - q.q[Q03] - q.q[Q04] - q.q[Q05] - q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15];


                m[M06] = q.q[Q07] - q.q[Q14] + q.q[Q08] - q.q[Q13];

                m[M07] =  - q.q[Q01] - q.q[Q02] - q.q[Q05] - q.q[Q06] + q.q[Q07] + q.q[Q14] + q.q[Q08] + q.q[Q13] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];

                m[M08] = q.q[Q09] - q.q[Q16] + q.q[Q10] - q.q[Q15];

                m[M09] = q.q[Q11] - q.q[Q18] + q.q[Q12] - q.q[Q17];

                m[M10] =  - q.q[Q01] - q.q[Q03] - q.q[Q02] - q.q[Q04] + q.q[Q09] + q.q[Q16] + q.q[Q10] + q.q[Q15] + q.q[Q11] + q.q[Q18] + q.q[Q12] + q.q[Q17];

                m[M11] =  - q.q[Q01] + q.q[Q02] + 2*q.q[Q07] - 2*q.q[Q14] - 2*q.q[Q08] + 2*q.q[Q13] - q.q[Q09] + q.q[Q16] + q.q[Q10] - q.q[Q15];

                m[M12] =  - q.q[Q03] + q.q[Q04] + 2*q.q[Q07] + 2*q.q[Q14] - 2*q.q[Q08] - 2*q.q[Q13] - q.q[Q11] + q.q[Q18] + q.q[Q12] - q.q[Q17];

                m[M13] =  - 3*q.q[Q01] + 3*q.q[Q02] + 3*q.q[Q09] - 3* q.q[Q16] - 3*q.q[Q10] + 3*q.q[Q15];

                m[M14] =  - 3*q.q[Q03] + 3*q.q[Q04] + 3*q.q[Q11] - 3*q.q[Q18] - 3*q.q[Q12] + 3*q.q[Q17];

                m[M15] =  - 2*q.q[Q05] + 2*q.q[Q06] + q.q[Q09] + q.q[Q16] - q.q[Q10] - q.q[Q15] + q.q[Q11] + q.q[Q18] - q.q[Q12] - q.q[Q17];

                m[M16] =  - 3*q.q[Q09] - 3*q.q[Q16] + 3*q.q[Q10] + 3*q.q[Q15] + 3*q.q[Q11] + 3*q.q[Q18] - 3*q.q[Q12] - 3*q.q[Q17];

                m[M17] = 0.5*q.q[Q01] + 0.5*q.q[Q03] + 0.5*q.q[Q02] + 0.5*q.q[Q04] - q.q[Q05] - q.q[Q06] - q.q[Q07] - q.q[Q14] - q.q[Q08] - q.q[Q13] + 0.5*q.q[Q09] + 0.5*q.q[Q16] + 0.5*q.q[Q10] + 0.5*q.q[Q15] + 0.5*q.q[Q11] + 0.5*q.q[Q18] + 0.5*q.q[Q12] + 0.5*q.q[Q17];

                m[M18] = 1.5*q.q[Q01] - 1.5*q.q[Q03] + 1.5*q.q[Q02] - 1.5*q.q[Q04] - 1.5*q.q[Q09] - 1.5* q.q[Q16] - 1.5* q.q[Q10] - 1.5* q.q[Q15] + 1.5*q.q[Q11] + 1.5*q.q[Q18] + 1.5*q.q[Q12] + 1.5*q.q[Q17];

                AF::writeMoments(*this, m, i, j, k);

            }
        }
    }



}//end of func




