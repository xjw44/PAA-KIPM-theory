import math 
import numpy as np
import data_sum_config
import matplotlib.pyplot as plt
import os
import pandas as pd
import diffuse_config
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import LogLocator, LogFormatter

# integrate diffusion
from scipy import integrate
import numpy as np

# plot tau_s 
from scipy.constants import hbar, electron_volt
import math
from scipy.constants import Boltzmann, e

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

plt.rc('font', size=16)  # Set the global default font size

target_l = 20 # mu_M

def diffusion_eff(lambda_d, lambda_a): 
    numer = lambda_d**2 * (lambda_a+lambda_d*math.tanh(1/2/lambda_d)) * math.sinh(1/lambda_d)
    denom1 = lambda_d*2*lambda_a*math.cosh(1/lambda_d)
    denom2 = (lambda_d**2+lambda_a**2)*math.sinh(1/lambda_d)
    eff = 2*numer/(denom1+denom2)
    # print(denom_1, denom_2)
    return eff

def lambda_func_abs(param_dict, l_over_array, w_over_array, l_abs, w_abs, l_over_org=5*1e-6): 
    d_diffuse = param_dict['d_diffuse']
    tau_decay = param_dict['tau_decay']
    array_scale_tau_trap = l_over_array*w_over_array

    tau_trap = param_dict['tau_trap']*w_abs*l_over_org/array_scale_tau_trap
    ld = np.sqrt(d_diffuse*tau_decay)
    la = np.sqrt(d_diffuse*tau_trap)
    lambda_d = ld/l_abs
    lambda_a = la/l_abs
    return lambda_d, lambda_a

def lambda_func_over(param_dict, l_over_array, w_over_array, w_kid): 
    d_diffuse = param_dict['d_diffuse_over']
    tau_decay = param_dict['tau_decay_over']

    tau_trap = param_dict['tau_trap_over']*w_over_array/w_kid
    ld = np.sqrt(d_diffuse*tau_decay)
    la = np.sqrt(d_diffuse*tau_trap)
    lambda_d = ld/l_over_array
    lambda_a = la/l_over_array
    return lambda_d, lambda_a

def get2DEff(param_dict_abs, param_dict_over, l_over_array, w_over_array, 
    l_abs=100*1e-6, w_abs=100*1e-6, w_kid=1e-6): 
    lambda_d_abs, lambda_a_array_abs = lambda_func_abs(param_dict_abs, 
        l_over_array, w_over_array, l_abs, w_abs)
    lambda_d_over_array, lambda_a_array_over = lambda_func_over(param_dict_over, 
        l_over_array, w_over_array, w_kid)

    # Empty array to store efficiencies
    eff_array = np.zeros_like(lambda_a_array_abs)
    eff_array_abs = np.zeros_like(lambda_a_array_abs)
    eff_array_over = np.zeros_like(lambda_a_array_abs)

    # Loop and fill
    for i in range(eff_array.shape[0]):
        for j in range(eff_array.shape[1]):
            lambda_a_abs = lambda_a_array_abs[i, j]
            abs_eff = diffusion_eff(lambda_d_abs, lambda_a_abs)

            lambda_d_over = lambda_d_over_array[i, j]
            lambda_a_over = lambda_a_array_over[i, j]
            over_eff = diffusion_eff(lambda_d_over, lambda_a_over)
            eff_array[i, j] = abs_eff*over_eff
            eff_array_abs[i, j] = abs_eff
            eff_array_over[i, j] = over_eff

    return eff_array, eff_array_abs, eff_array_over

def plot2DEff(eff_array, l_over_array, w_over_array, plot_dir, 
    param_dict_abs, param_dict_over, eff_array_abs, eff_array_over): 
    plt.figure(figsize=(8, 6))
    im = plt.pcolormesh(l_over_array*1e6, w_over_array*1e6, 
    eff_array*100, shading='auto')  # convert to μm for display

    # Find max efficiency and its location
    eff_max = np.max(eff_array*100)
    max_idx = np.unravel_index(np.argmax(eff_array), eff_array.shape)
    l_over_max = l_over_array[max_idx]
    w_over_max = w_over_array[max_idx]
    eff_max_abs = eff_array_abs[max_idx]*100
    eff_max_over = eff_array_over[max_idx]*100

    # Add colorbar and labels
    cbar = plt.colorbar(im, label=r"Efficiency (\%)")
    plt.xlabel(r"$l_{{over}}$ ($\mathrm{{\mu m}}$)")
    plt.ylabel(r"$w_{{over}}$ ($\mathrm{{\mu m}}$)")
    plt.title("Combined diffusion efficiency")

    # mark max
    long_label = rf"{eff_max:.1f}%@"+"\n"+ \
        rf"$l_{{over}}$={l_over_max*1e6:.1f} $\mathrm{{\mu m}}$"+"\n"+ \
        rf"$w_{{over}}$={w_over_max*1e6:.1f} $\mathrm{{\mu m}}$"+"\n"+ \
        fr"$\tau_{{decay}}^{{abs}}=${param_dict_abs["tau_decay"]*1e6:.2g} $\mathrm{{\mu}}s$"+"\n"+ \
        fr"$\tau_{{decay}}^{{over}}=${param_dict_over["tau_decay_over"]*1e6:.2g} $\mathrm{{\mu}}s$"+"\n"+ \
        fr"$\tau_{{trap,\,0}}^{{abs}}=${param_dict_abs["tau_trap"]*1e6:.3g} $\mathrm{{\mu}}s$"+"\n"+ \
        fr"$\tau_{{trap,\,0}}^{{over}}=${param_dict_over["tau_trap_over"]*1e6:.3g} $\mathrm{{\mu}}s$"+"\n"+ \
        fr"$D_{{abs}}$={param_dict_abs["d_diffuse"]:.2g} $\mathrm{{m^2}}$/s"+"\n"+ \
        fr"$D_{{over}}$={param_dict_over["d_diffuse_over"]:.2g} $\mathrm{{m^2}}$/s"+"\n"+ \
        rf"$\eta_{{d,\,abs}}$={eff_max_abs:.1f}%"+"\n"+ \
        rf"$\eta_{{d,\,over}}$={eff_max_over:.1f}%"
    plt.plot(l_over_max*1e6, w_over_max*1e6, 'ro', label=long_label)  # red circle

    # Add contour at 80% and 60% of max
    contour_levels = [0.8*eff_max, 0.9*eff_max]
    contours = plt.contour(l_over_array * 1e6, w_over_array * 1e6, eff_array * 100,
                           levels=contour_levels, colors='black', linewidths=1.0)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f%%")

    # Save figure
    save_dir = os.path.dirname(plot_dir)
    if save_dir:  # avoid error if save_path is just a filename
        os.makedirs(save_dir, exist_ok=True)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(plot_dir+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir+".png", dpi=300, bbox_inches='tight')
    plt.close()

    return eff_array

def plotTau_a(plot_dir, tau_a_ref_mus, ref_thickness_nm): 
    plt.figure(figsize=(8, 6))
    thickness = np.linspace(0, 1000, 1000) # nm
    tau_a = tau_a_ref_mus*(thickness/ref_thickness_nm)**3
    plt.plot(thickness, tau_a)

    # Mark vertical lines at specific thicknesses
    for t_val in [100, 600]:
        tau_val = tau_a_ref_mus * (t_val / ref_thickness_nm)**3
        plt.axvline(t_val, color='red', linestyle='--')
        plt.plot(t_val, tau_val, 'ro', label=rf'$\tau_{{trap}}$={tau_val:.2f} $\mu$s @ {t_val:.1f} nm')

    # Mark each tau_target with vertical & horizontal lines
    for tau_target in [0.35]:
        thickness_target = ref_thickness_nm * (tau_target / tau_a_ref_mus)**(1/3)
        plt.plot(thickness_target, tau_target, 'bo', 
            label=rf'$\tau_{{trap}}$={tau_target:.2f} $\mu$s @ {thickness_target:.1f} nm')
        plt.axhline(tau_target, color='blue', linestyle=':')

    plt.xlabel('thickness (nm)')
    plt.ylabel(r'$\tau_{trap}$ ($\mu$s)')
    plt.title(r'expected $\tau_{trap}$')
    plt.grid(True)
    
    # Save figure
    save_dir = os.path.dirname(plot_dir)
    if save_dir:  # avoid error if save_path is just a filename
        os.makedirs(save_dir, exist_ok=True)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(plot_dir+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir+".png", dpi=300, bbox_inches='tight')
    plt.close()

    return True

def compute_tau_s(omega, delta, z1_zero=1.43, b=317):
    """
    Compute the tau_s lifetime.

    Parameters:
        delta_abs (float): Absorber superconducting gap [eV or J]
        delta_kid (float): KID superconducting gap [eV or J]
        tau_factor (float): Scaling factor [s·eV^3]
        z1_zero = 1.43 # renormalization factor 
        # al_b = 0.317*1e-3 # mev**-2 
        al_b = 317 # ev**-2 

    Returns:
        float: tau_s in seconds
    """
    hbar_ev = hbar / electron_volt #ev*s
    kb_ev = Boltzmann / e #ev/k

    tau_factor = 3*z1_zero*hbar_ev/(2*math.pi*b) # mev^3/s

    # t_c = 1.2 # k 
    # tau_0 = z1_zero*hbar_ev/(2*math.pi*al_b)/(kb_ev*t_c)**3

    tau_s = tau_factor/omega**3*(1-(delta/omega)**2)**(-3/2) #s
    return tau_s

def plotTau_s(plot_dir, delta_al=0.2*1e-3, delta_hf=0.04*1e-3, tau_s_target_list=[12.5*1e-6, 0.058*1e-6]): 
    # delta_al = 0.2*1e-3 # eV
    # delta_hf = 0.04*1e-3 # ev
    delta_over = np.linspace(delta_hf*(1+1e-6), delta_al*(1-1e-6), 1000) # ev

    plt.figure(figsize=(8, 6))
    tau_s_abs_over = compute_tau_s(delta=delta_over, omega=delta_al)
    tau_s_over_kid = compute_tau_s(delta=delta_hf, omega=delta_over)
    plt.plot(delta_over*1e3, tau_s_abs_over*1e6, label=r'$\tau_{s}(abs\rightarrow overlap)$')
    plt.plot(delta_over*1e3, tau_s_over_kid*1e6, label=r'$\tau_{s}(overlap\rightarrow kid)$')
    plt.yscale('log')

    # Find delta_over where tau_s crosses target
    def find_crossing(xvals, yvals, target):
        diff = np.abs(yvals - target)
        idx = np.argmin(diff)
        print(np.min(diff))
        return xvals[idx]

    # Draw horizontal lines and annotate for each target tau_s
    for tau_s_target in tau_s_target_list:
        delta_over_abs_over = find_crossing(delta_over, tau_s_abs_over, tau_s_target)
        delta_over_over_kid = find_crossing(delta_over, tau_s_over_kid, tau_s_target)

        # Horizontal line
        long_label = fr'$\tau_s$={tau_s_target*1e6:.3g} $\mu$s'+'\n'+\
                     fr'@$\Delta_{{over}}(abs\rightarrow overlap)$={delta_over_abs_over*1e3:.3f} meV'+'\n'+\
                     fr'@$\Delta_{{over}}(overlap\rightarrow kid)$={delta_over_over_kid*1e3:.3f} meV'
        plt.axhline(tau_s_target*1e6, color='gray', linestyle=':', label=long_label)

    # Compute global minimum tau_s
    min_taus_abs_over = np.min(tau_s_abs_over)
    min_taus_over_kid = np.min(tau_s_over_kid)

    # Global minimum
    min_taus_global = min(min_taus_over_kid, min_taus_abs_over)

    # Plot horizontal line at global minimum
    plt.axhline(min_taus_global*1e6, color='black', linestyle='--', linewidth=1.0,
                label=fr'Min $\tau_s$ = {min_taus_global*1e6:.3g} $\mu$s')

    plt.xlabel(r'$\Delta_{over}$ (meV)')
    plt.ylabel(r'$\tau_{s}$ ($\mu$s)')
    plt.title(r'expected $\tau_{s}$')
    # Major ticks at 10^0, 10^1, ...
    # plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    # plt.gca().yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=True))  # labels all major ticks like 10^0, 10^1

    # Minor ticks at 2–9 in each decade
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    plt.grid(True)
    
    # Save figure
    save_dir = os.path.dirname(plot_dir)
    if save_dir:  # avoid error if save_path is just a filename
        os.makedirs(save_dir, exist_ok=True)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(plot_dir+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir+".png", dpi=300, bbox_inches='tight')
    plt.close()

    return True

def main():
    # #################################################### 
    # #################################################### 
    # #################################################### 7/7/2025 nominal comb:)) 
    # # Define ranges (in meters)
    # l_over_vals = np.linspace(1e-6, 40e-6, 100)  # 1 μm to 40 μm
    # w_over_vals = np.linspace(1e-6, 100e-6, 100)  # 1 μm to 100 μm

    # # Create 2D meshgrid
    # l_over, w_over = np.meshgrid(l_over_vals, w_over_vals)  # L and W both have shape (100, 100)
    # print(l_over, w_over)

    # title = 'comb_diffusion'
    # plot_dir = "output/2025-7-7-diffusion-eff-second/"
    # eff_array, eff_array_abs, eff_array_over = get2DEff(diffuse_config.param_dict_abs, 
    #     diffuse_config.param_dict_over, l_over, w_over)
    # plot2DEff(eff_array, l_over, w_over, plot_dir+title, 
    #     diffuse_config.param_dict_abs, diffuse_config.param_dict_over, eff_array_abs, eff_array_over)

    #################################################### 
    #################################################### 
    #################################################### 7/8/2025 check tau more! 
    title = 'tau_a_expected'
    plot_dir = "output/2025-7-8-combdeff-taus/"
    plotTau_a(plot_dir+title, diffuse_config.param_dict_abs["tau_trap"]*1e6, 600)
    title = 'tau_s_expected'
    plotTau_s(plot_dir+title)
    title = 'tau_s_expected_w'
    plotTau_s(plot_dir+title, delta_hf=0.015*1e-3)

    ########################################## tau_s config 
    delta_al = 0.2*1e-3 # eV
    delta_hf = 0.04*1e-3 # ev
    delta_over = (delta_hf+delta_al)/2 # mev

    tau_s_overlap = compute_tau_s(delta=delta_over, omega=delta_al)
    tau_s_kid = compute_tau_s(delta=delta_hf, omega=delta_over)
    print(delta_over, tau_s_overlap*6**3, tau_s_overlap, tau_s_kid)

    # Define ranges (in meters)
    l_over_vals = np.linspace(1e-6, 40e-6, 100)  # 1 μm to 40 μm
    w_over_vals = np.linspace(1e-6, 100e-6, 100)  # 1 μm to 100 μm

    # Create 2D meshgrid
    l_over, w_over = np.meshgrid(l_over_vals, w_over_vals)  # L and W both have shape (100, 100)

    # title = 'large_tau_trap_over'
    title = 'large_tau_decay'
    eff_array, eff_array_abs, eff_array_over = get2DEff(diffuse_config.param_dict_abs, 
        diffuse_config.param_dict_over, l_over, w_over)
    plot2DEff(eff_array, l_over, w_over, plot_dir+title, 
        diffuse_config.param_dict_abs, diffuse_config.param_dict_over, eff_array_abs, eff_array_over)


if __name__ == "__main__":
    main()
