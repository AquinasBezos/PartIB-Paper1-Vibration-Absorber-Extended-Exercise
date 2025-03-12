#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from parallel import MLKF_2dof, freq_response, time_response, last_nonzero

def sweep_inertance(m1, l1, k1, f1, m2, l2, k2, f2, b1_range, b2_range, hz_range, sec_max):
    """
    Sweep through ranges of inertance values (b1 and b2) and analyze system response
    
    Returns:
        results: Dictionary containing peak amplitudes and settling times for each b1, b2 combination
    """
    # Create frequency and time arrays
    hz = np.linspace(hz_range[0], hz_range[1], 1001)
    sec = np.linspace(0, sec_max, 1001)
    
    # Initialize result arrays
    b1_values = np.linspace(b1_range[0], b1_range[1], b1_range[2])
    b2_values = np.linspace(b2_range[0], b2_range[1], b2_range[2])
    
    # Initialize result matrices
    shape = (len(b1_values), len(b2_values))
    peak_amp_m1 = np.zeros(shape)
    peak_amp_m2 = np.zeros(shape)
    settle_time_m1 = np.zeros(shape)
    settle_time_m2 = np.zeros(shape)
    peak_freq_m1 = np.zeros(shape)
    peak_freq_m2 = np.zeros(shape)
    
    # Iterate through all combinations of b1 and b2
    for i, b1 in enumerate(b1_values):
        for j, b2 in enumerate(b2_values):
            # Calculate system matrices
            M, L, K, F = MLKF_2dof(m1, b1, l1, k1, f1, m2, b2, l2, k2, f2)
            
            # Get frequency response
            f_response = freq_response(hz * 2*np.pi, M, L, K, F)
            f_amplitude = np.abs(f_response)
            
            # Get peak amplitudes and frequencies
            m1_peak_idx = np.argmax(f_amplitude[:, 0])
            m2_peak_idx = np.argmax(f_amplitude[:, 1])
            peak_amp_m1[i, j] = f_amplitude[m1_peak_idx, 0]
            peak_amp_m2[i, j] = f_amplitude[m2_peak_idx, 1]
            peak_freq_m1[i, j] = hz[m1_peak_idx]
            peak_freq_m2[i, j] = hz[m2_peak_idx]
            
            # Get time response
            t_response = time_response(sec, M, L, K, F)
            
            # Calculate settling time (time to reach within 2% of equilibrium)
            equilib = np.abs(freq_response([0], M, L, K, F))[0]  # Response at 0 Hz
            toobig = abs(100 * (t_response - equilib) / equilib) >= 2
            lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)
            
            settle_time_m1[i, j] = sec[lastbig[0]]
            settle_time_m2[i, j] = sec[lastbig[1]]
    
    results = {
        'b1_values': b1_values,
        'b2_values': b2_values,
        'peak_amp_m1': peak_amp_m1,
        'peak_amp_m2': peak_amp_m2,
        'settle_time_m1': settle_time_m1,
        'settle_time_m2': settle_time_m2,
        'peak_freq_m1': peak_freq_m1,
        'peak_freq_m2': peak_freq_m2
    }
    
    return results

def plot_results(results):
    """Plot the results of the inertance sweep analysis"""
    b1_values = results['b1_values']
    b2_values = results['b2_values']
    
    # Create a meshgrid for 3D plotting
    B1, B2 = np.meshgrid(b1_values, b2_values, indexing='ij')
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # First row: Peak amplitude plots
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(B1, B2, results['peak_amp_m1'], cmap='viridis')
    ax1.set_title('Peak Amplitude - Mass 1')
    ax1.set_xlabel('Inertance b1')
    ax1.set_ylabel('Inertance b2')
    ax1.set_zlabel('Amplitude (m)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(B1, B2, results['peak_amp_m2'], cmap='viridis')
    ax2.set_title('Peak Amplitude - Mass 2')
    ax2.set_xlabel('Inertance b1')
    ax2.set_ylabel('Inertance b2')
    ax2.set_zlabel('Amplitude (m)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # Second row: Settling time plots
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(B1, B2, results['settle_time_m1'], cmap='plasma')
    ax3.set_title('Settling Time - Mass 1')
    ax3.set_xlabel('Inertance b1')
    ax3.set_ylabel('Inertance b2')
    ax3.set_zlabel('Time (s)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(B1, B2, results['settle_time_m2'], cmap='plasma')
    ax4.set_title('Settling Time - Mass 2')
    ax4.set_xlabel('Inertance b1')
    ax4.set_ylabel('Inertance b2')
    ax4.set_zlabel('Time (s)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    # Generate contour plots for another view
    fig2 = plt.figure(figsize=(15, 10))
    
    ax5 = fig2.add_subplot(2, 2, 1)
    cont1 = ax5.contourf(B1, B2, results['peak_amp_m1'], cmap='viridis', levels=20)
    ax5.set_title('Peak Amplitude - Mass 1 (Contour)')
    ax5.set_xlabel('Inertance b1')
    ax5.set_ylabel('Inertance b2')
    fig2.colorbar(cont1, ax=ax5)
    
    ax6 = fig2.add_subplot(2, 2, 2)
    cont2 = ax6.contourf(B1, B2, results['peak_amp_m2'], cmap='viridis', levels=20)
    ax6.set_title('Peak Amplitude - Mass 2 (Contour)')
    ax6.set_xlabel('Inertance b1')
    ax6.set_ylabel('Inertance b2')
    fig2.colorbar(cont2, ax=ax6)
    
    ax7 = fig2.add_subplot(2, 2, 3)
    cont3 = ax7.contourf(B1, B2, results['settle_time_m1'], cmap='plasma', levels=20)
    ax7.set_title('Settling Time - Mass 1 (Contour)')
    ax7.set_xlabel('Inertance b1')
    ax7.set_ylabel('Inertance b2')
    fig2.colorbar(cont3, ax=ax7)
    
    ax8 = fig2.add_subplot(2, 2, 4)
    cont4 = ax8.contourf(B1, B2, results['settle_time_m2'], cmap='plasma', levels=20)
    ax8.set_title('Settling Time - Mass 2 (Contour)')
    ax8.set_xlabel('Inertance b1')
    ax8.set_ylabel('Inertance b2')
    fig2.colorbar(cont4, ax=ax8)
    
    plt.tight_layout()
    
    # Optional: Plot resonant frequencies
    fig3 = plt.figure(figsize=(12, 5))
    
    ax9 = fig3.add_subplot(1, 2, 1, projection='3d')
    surf5 = ax9.plot_surface(B1, B2, results['peak_freq_m1'], cmap='coolwarm')
    ax9.set_title('Resonant Frequency - Mass 1')
    ax9.set_xlabel('Inertance b1')
    ax9.set_ylabel('Inertance b2')
    ax9.set_zlabel('Frequency (Hz)')
    fig3.colorbar(surf5, ax=ax9, shrink=0.5, aspect=5)
    
    ax10 = fig3.add_subplot(1, 2, 2, projection='3d')
    surf6 = ax10.plot_surface(B1, B2, results['peak_freq_m2'], cmap='coolwarm')
    ax10.set_title('Resonant Frequency - Mass 2')
    ax10.set_xlabel('Inertance b1')
    ax10.set_ylabel('Inertance b2')
    ax10.set_zlabel('Frequency (Hz)')
    fig3.colorbar(surf6, ax=ax10, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Inertance Sweep Analysis for 2DOF System')
    
    # System parameters (with defaults from parallel.py)
    parser.add_argument('--m1', type=float, default=3.94, help='Mass 1 (kg)')
    parser.add_argument('--l1', type=float, default=1.73, help='Damping 1 (N·s/m)')
    parser.add_argument('--k1', type=float, default=2106.5, help='Spring 1 (N/m)')
    parser.add_argument('--f1', type=float, default=2.4, help='Force 1 (N)')
    
    parser.add_argument('--m2', type=float, default=0.15, help='Mass 2 (kg)')
    parser.add_argument('--l2', type=float, default=0.9, help='Damping 2 (N·s/m)')
    parser.add_argument('--k2', type=float, default=80.2, help='Spring 2 (N/m)')
    parser.add_argument('--f2', type=float, default=0, help='Force 2 (N)')
    
    # Inertance sweep parameters
    parser.add_argument('--b1-range', type=float, nargs=3, default=[-3.5, 0, 20],
                        help='Range for b1 [min, max, steps]')
    parser.add_argument('--b2-range', type=float, nargs=3, default=[-0.005, 0.05, 20],
                        help='Range for b2 [min, max, steps]')
    
    # Analysis parameters
    parser.add_argument('--hz-range', type=float, nargs=2, default=[0, 5],
                        help='Frequency range [min, max] (Hz)')
    parser.add_argument('--sec-max', type=float, default=30,
                        help='Maximum time for time-domain analysis (s)')
    
    args = parser.parse_args()
    
    # Convert number of steps to integer
    b1_range = [args.b1_range[0], args.b1_range[1], int(args.b1_range[2])]
    b2_range = [args.b2_range[0], args.b2_range[1], int(args.b2_range[2])]
    
    print(f"Sweeping b1 from {b1_range[0]} to {b1_range[1]} with {b1_range[2]} steps")
    print(f"Sweeping b2 from {b2_range[0]} to {b2_range[1]} with {b2_range[2]} steps")
    print("This may take a while depending on the number of steps...")
    
    # Run the analysis
    results = sweep_inertance(
        args.m1, args.l1, args.k1, args.f1,
        args.m2, args.l2, args.k2, args.f2,
        b1_range, b2_range,
        args.hz_range, args.sec_max
    )
    
    # Plot the results
    plot_results(results)

if __name__ == '__main__':
    main()