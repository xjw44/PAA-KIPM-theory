import math 
import numpy as np
import data_sum_config
import matplotlib.pyplot as plt
import os
import pandas as pd

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
        rf"$l_{{over}}$={l_over_max*1e6:.1f}$\mathrm{{\mu m}}$"+"\n"+ \
        rf"$w_{{over}}$={w_over_max*1e6:.1f}$\mathrm{{\mu m}}$"+"\n"+ \
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

def targetEff(df, target_l=100):
    """
    Given a DataFrame with 'l_abs_list' and 'eff_list' columns,
    returns a Series of efficiencies (in %) at l_abs ≈ 100 nm for all rows.
    # target_l = 100 µm
    """

    def get_eff(row):
        l_abs_um = np.array(row["l_abs_list"]) * 1e6     # convert m to µm
        eff_percent = np.array(row["eff_list"]) * 100    # convert to %
        idx = np.argmin(np.abs(l_abs_um - target_l))
        return eff_percent[idx]

    return df.apply(get_eff, axis=1)

def loopThroughEffs(ax, df, thickness=False): 
    colors = ['red', 'blue', 'pink']
    df = df.reset_index(drop=True)
    df["eff_at_target"] = targetEff(df, target_l)

    for idx, row in df.iterrows():
        if not thickness: 
            label_long = fr"$\tau_{{decay}}=${row["tau_decay"]*1e6:.2g} $\mathrm{{\mu}}s$"+"\n"+ \
                fr"$\tau_{{trap}}=${row["tau_trap"]*1e6:.3g} $\mathrm{{\mu}}s$"+"\n"+ \
                fr"D={row["d_diffuse"]:.2g} $\mathrm{{m^2}}$/s"+"\n"+ \
                fr"$\eta_{{d}}$={row["eff_at_target"]:.1f}$\%$"+"\n" \
                fr"$w_{{over}}$={row["w_over"]*1e6:.0f} $\mathrm{{\mu m}}$"+"\n"\
                # fr"$l_{{over}}$={row["l_over"]*1e6:.0f} $\mathrm{{\mu m}}$"+"\n"
            ax.plot(row["l_abs_list"]*1e6, row["eff_list"]*100, marker='o', 
                label=label_long, 
                color=colors[idx])
        else: 
            if isinstance(row["thickness"], str):
                thickness = eval(row["thickness"])
            else: 
                thickness = row["thickness"]
            label_long=fr"$l_{{a}}=${row["l_a_scaled"]*1e6:.1f} $\mathrm{{\mu}}m$"+"\n"+ \
            fr"$l_{{d}}=${row["l_d_scaled"]*1e6:.1f} $\mathrm{{\mu}}m$"+"\n"+ \
            fr"$t_{{abs}}=${thickness*1e9:.0f} nm"+"\n"+\
            fr"$\eta_{{d}}$={row["eff_at_target"]:.1f}$\%$"+"\n"
            ax.plot(row["l_abs_list"]*1e6, row["eff_list"]*100, marker='o', 
            label=label_long, 
            color=colors[idx])
    ax.set_xlabel(r'$l_{abs}$ ($\mu$m)')
    ax.set_ylabel('QP diffusion efficiency (%)')
    ax.set_title('QP diffusion')
    ax.grid(True)
    # ax.axvline(x=target_l, label=rf'$l_{{abs}}$={target_l} $\mu$m', color='black')
    ax.axvline(x=target_l, label=rf'$l_{{over}}$={target_l} $\mu$m', color='black')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plotDiffusionEff(d_diffuse, tau_decay, tau_trap, l_abs_list, plot_dir,
    thickness=600*1e-9, # m
    w_abs=100*1e-6, # m
    l_over=5*1e-6, # m
    ): 
    vary_list = [0.1, 10]
    add_kid_length = [10*1e-6, 100*1e-6] # m
    vary_thickness = [100*1e-9, 1000*1e-9]
    vary_overlap_w = [1*1e-6, 20*1e-6]

    df_default = pd.DataFrame([{"d_diffuse": d_diffuse, "tau_decay": tau_decay, 
    "tau_trap": tau_trap, "l_abs_list": l_abs_list, 
    "thickness": thickness, "w_over": w_abs,  "l_over": l_over, 
    "l_d": np.sqrt(d_diffuse*tau_decay), 
    "l_a": np.sqrt(d_diffuse*tau_trap), "eff_list": 0, 
    "l_d_scaled": 0, 
    "l_a_scaled": 0}])

    scaled_dfs = []
    # for vary in vary_list:
    #     for var in ["d_diffuse", "tau_decay", "tau_trap"]:
    #         df_scaled = df_default.copy()
    #         df_scaled[var] = df_scaled[var] * vary
    #         scaled_dfs.append(df_scaled)

    for l_kid_short in add_kid_length:
        df_scaled = df_default.copy()
        df_scaled["w_over"] = l_kid_short
        scaled_dfs.append(df_scaled)

    # for l_over_short in vary_overlap_w:
    #     df_scaled = df_default.copy()
    #     df_scaled["l_over"] = l_over_short
    #     scaled_dfs.append(df_scaled)

    # for thick in vary_thickness:
    #     df_scaled = df_default.copy()
    #     df_scaled["thickness"] = thick
    #     scaled_dfs.append(df_scaled)

    df_scaled = pd.concat(scaled_dfs, ignore_index=True)
    df_all_effs = pd.concat([df_scaled, df_default], ignore_index=True)
    df_all_effs['eff_list'] = df_all_effs['eff_list'].astype(object)
    for idx, row in df_all_effs.iterrows():
        eff_list_row, l_d_scaled, l_a_scaled = getListEff(row["d_diffuse"], 
            row["tau_decay"], row["tau_trap"], row["l_abs_list"], l_d=row["l_d"], l_a=row["l_a"])
        if row["w_over"] !=w_abs:
            # eff_list_row, l_d_scaled, l_a_scaled = getListEff(row["d_diffuse"], 
            #     row["tau_decay"], row["tau_trap"]/row["w_over"]*w_abs, row["l_abs_list"])
            eff_list_row, l_d_scaled, l_a_scaled = getListEff(row["d_diffuse"], 
                row["tau_decay"], row["tau_trap"]*row["w_over"]/w_abs, row["l_abs_list"])
        elif row["thickness"] !=thickness:
            eff_list_row, l_d_scaled, l_a_scaled = getListEff(row["d_diffuse"], 
                row["tau_decay"], row["tau_trap"], row["l_abs_list"], 
                uselambdas=True, 
                thickness=row["thickness"], l_d=row["l_d"], l_a=row["l_a"])
        elif row["l_over"] !=l_over:
            eff_list_row, l_d_scaled, l_a_scaled = getListEff(row["d_diffuse"], 
                row["tau_decay"], row["tau_trap"]/row["l_over"]*l_over, row["l_abs_list"])
        df_all_effs.at[idx, 'eff_list'] = eff_list_row
        df_all_effs['l_d_scaled'] = df_all_effs['l_d_scaled'].astype(float)
        df_all_effs['l_a_scaled'] = df_all_effs['l_a_scaled'].astype(float)
        df_all_effs.at[idx, 'l_d_scaled'] = l_d_scaled
        df_all_effs.at[idx, 'l_a_scaled'] = l_a_scaled
    print(df_all_effs)

    df_diffuse = df_all_effs[(df_all_effs['tau_decay'] == tau_decay) 
        & (df_all_effs['tau_trap'] == tau_trap)
        & (df_all_effs['w_over'] == w_abs)
        & (df_all_effs['thickness'] == thickness)
        & (df_all_effs['l_over'] == l_over)]
    df_decay = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_trap'] == tau_trap)
        & (df_all_effs['w_over'] == w_abs)
        & (df_all_effs['thickness'] == thickness)
        & (df_all_effs['l_over'] == l_over)]
    df_trap = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_decay'] == tau_decay)
        & (df_all_effs['w_over'] == w_abs)
        & (df_all_effs['thickness'] == thickness)
        & (df_all_effs['l_over'] == l_over)]
    df_trans = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_decay'] == tau_decay)
        & (df_all_effs['tau_trap'] == tau_trap)
        & (df_all_effs['thickness'] == thickness)
        & (df_all_effs['l_over'] == l_over)]
    df_thickness = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_decay'] == tau_decay)
        & (df_all_effs['tau_trap'] == tau_trap)
        & (df_all_effs['w_over'] == w_abs)
        & (df_all_effs['l_over'] == l_over)]
    df_over = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_decay'] == tau_decay)
        & (df_all_effs['tau_trap'] == tau_trap)
        & (df_all_effs['thickness'] == thickness)
        & (df_all_effs['w_over'] == w_abs)]

    fig, axes = plt.subplots(2, 3, figsize=(30, 12))
    # loopThroughEffs(axes[0,0], df_diffuse)
    # loopThroughEffs(axes[0,1], df_decay)
    # loopThroughEffs(axes[1,0], df_trap)
    loopThroughEffs(axes[1,1], df_trans)
    # loopThroughEffs(axes[1,2], df_over)
    # loopThroughEffs(axes[0,2], df_thickness, thickness=True)

    # Save figure
    save_dir = os.path.dirname(plot_dir)
    if save_dir:  # avoid error if save_path is just a filename
        os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_dir+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir+".png", dpi=300, bbox_inches='tight')
    plt.close()

    return True

def main():
    ld = 300*1e-6 # m
    ld_over = 50*1e-6 # m

    d_diffuse = 0.02 #m^2/s
    d_diffuse_over = d_diffuse/6

    tau_decay = ld**2/d_diffuse #s
    tau_decay_over = ld_over**2/d_diffuse_over #s
    
    la = 500*1e-6 # m
    la_over = 13.9*1e-6 # m
    tau_trap = la**2/d_diffuse # m/s
    tau_trap_over = la_over**2/d_diffuse_over # m/s

    # Define ranges (in meters)
    l_over_vals = np.linspace(1e-6, 40e-6, 100)  # 1 μm to 40 μm
    w_over_vals = np.linspace(1e-6, 100e-6, 100)  # 1 μm to 100 μm

    # Create 2D meshgrid
    l_over, w_over = np.meshgrid(l_over_vals, w_over_vals)  # L and W both have shape (100, 100)

    print(tau_decay, tau_trap)
    print(d_diffuse_over, tau_decay_over, tau_trap_over)
    print(l_over, w_over)

    param_dict_abs = {"d_diffuse": d_diffuse, 
    "tau_trap": tau_trap,
    "tau_decay": tau_decay
    }
    param_dict_over = {"d_diffuse_over": d_diffuse_over, 
    "tau_trap_over": tau_trap_over,
    "tau_decay_over": tau_decay_over
    }

    title = 'comb_diffusion'
    plot_dir = "output/2025-7-7-diffusion-eff-second/"
    eff_array, eff_array_abs, eff_array_over = get2DEff(param_dict_abs, param_dict_over, l_over, w_over)
    plot2DEff(eff_array, l_over, w_over, plot_dir+title, 
        param_dict_abs, param_dict_over, eff_array_abs, eff_array_over)


if __name__ == "__main__":
    main()
