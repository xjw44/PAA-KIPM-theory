import math 
import numpy as np
import data_sum_config
import matplotlib.pyplot as plt
import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

plt.rc('font', size=16)  # Set the global default font size

target_l = 5 # mu_M

# def diffusion_eff(alpha, beta): 
#     numer = 2*math.sinh(alpha)*(beta+math.tanh(alpha))
#     denom_1 = 2*alpha*beta*math.cosh(alpha)
#     denom_2 = alpha*(1+beta**2)*math.sinh(alpha)
#     eff = numer/(denom_1+denom_2)
#     # print(denom_1, denom_2)
#     return eff

def diffusion_eff(lambda_d, lambda_a): 
    numer = lambda_d**2 * (lambda_a+lambda_d*math.tanh(1/2/lambda_d)) * math.sinh(1/lambda_d)
    denom1 = lambda_d*2*lambda_a*math.cosh(1/lambda_d)
    denom2 = (lambda_d**2+lambda_a**2)*math.sinh(1/lambda_d)
    eff = 2*numer/(denom1+denom2)
    # print(denom_1, denom_2)
    return eff

def getTrapeff(d_diffuse, tau_decay, tau_trap, t_abs, plot_name, 
    l_kid=1*1e-6, # m
    l_abs=100*1e-6, # m
    ): 
    eff = (1 + np.sqrt(tau_trap)/tau_decay*t_abs/np.sqrt(d_diffuse))**(-1)
    if plot_name=="sideway":
        eff = (1 + np.sqrt(tau_trap)/tau_decay*t_abs*l_abs/l_kid/np.sqrt(d_diffuse))**(-1)

    # print(denom_1, denom_2)
    return eff

def lambda_func(l_abs, d_diffuse, tau_decay, tau_trap): 
    ld = np.sqrt(d_diffuse*tau_decay)
    la = np.sqrt(d_diffuse*tau_trap)
    lambda_d = ld/l_abs
    lambda_a = la/l_abs
    return lambda_d, lambda_a

# def alpha_func(l, diffuse, tau_decay): 
#     '''
#     l: length (m)
#     diffuse: diffusion constant (m**2/s)
#     tau_decay: decay time (s)
#     alpha: unitless 
#     '''
#     alpha = l/np.sqrt(diffuse*tau_decay)
#     return alpha

# def beta_func(tau_trap, tau_decay): 
#     '''
#     tau_trap: the time it takes to trap a qp (s)
#     tau_decay: the time it takes to decay for a qp (s)
#     beta: unitless 
#     '''
#     beta = np.sqrt(tau_trap/tau_decay)
#     return beta

def diffuse_constant_plot(plot_dir, config): 
    # Extract data into a list of dicts
    records = []
    for key, value in config.items():
        record = {
            "label": key,
            "diffusion_constant": value.get("diffusion_constant"),
            "diffusion_constant_err": value.get("diffusion_constant_err", None),
            "unit": value.get("unit"),
            "temp_K": value.get("temp_K"),
            "material": value.get("material"),
            "film_thickness_m": value.get("film_thickness_m"),
            "method": value.get("method"),
            "source": value.get("source"),
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame.from_records(records)

    fig = plt.figure(figsize=(8,6))
    cmap = plt.cm.inferno
    num_rows = len(df)+1
    colors = cmap(np.linspace(0, 1, num_rows))

    for idx, row in df.iterrows():
        x = idx  # simple integer index for x-axis
        y = row['diffusion_constant']
        yerr = row['diffusion_constant_err'] if pd.notna(row['diffusion_constant_err']) else 0
        temp_mk = float(row['temp_K']) * 1e3 if row['temp_K'] != "NA" else np.nan
        thickness_nm = float(row['film_thickness_m']) * 1e9 if row['film_thickness_m'] != "NA" else np.nan
        y_scaled_temp = float(row['diffusion_constant'])*np.sqrt(35/temp_mk) if row['temp_K'] != "NA" else np.nan
        y_scaled_temp_thickness = y_scaled_temp/(thickness_nm/600) if row['film_thickness_m'] != "NA" else np.nan
    
        # Plot with error bar if available
        plt.errorbar(x, y, yerr=yerr, fmt='o', 
            label=fr'{row['diffusion_constant']:.4f} $\mathrm{{m^2}}$/s;'+
            f"\n{row['material']}@{temp_mk:.0f} mK;\n"+
            f"film thickness (nm): {thickness_nm:.0f}", color=colors[idx])
        if not np.isnan(y_scaled_temp_thickness):
            plt.errorbar(x, y_scaled_temp_thickness, yerr=yerr, fmt='s', 
                label=fr"scaled to {y_scaled_temp_thickness:.4f} $\mathrm{{m^2}}$/s", color=colors[idx])

    # Customize plot
    plt.xticks(ticks=range(len(df)), labels=df['label'], rotation=45, ha='right')
    plt.ylabel(r'Diffusion Constant ($\mathrm{m^2}$/s)')
    plt.title('Diffusion Constants from Various Sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir+"/diffuse_constant_sum"+".pdf")
    plt.close()

def getListEff(d_diffuse, tau_decay, tau_trap, l_abs_list, uselambdas=False, 
    thickness=0, l_d=0, l_a=0, thickness_default=600*1e-9): 
    # alpha_list = alpha_func(l_abs_list, d_diffuse, tau_decay)
    # beta = beta_func(tau_trap, tau_decay)
    lambda_d_list, lambda_a_list = lambda_func(l_abs_list, d_diffuse, tau_decay, tau_trap)
    l_d_scaled = l_d
    l_a_scaled = l_a
    if uselambdas:
        l_d_scaled = l_d*thickness/thickness_default
        l_a_scaled = l_a*(thickness**2)/(thickness_default**2)
        lambda_d_list = l_d_scaled/l_abs_list
        lambda_a_list = l_a_scaled/l_abs_list

    eff_list = []
    # for alpha in alpha_list: 
    for idx, lambda_d in enumerate(lambda_d_list): 
        eff = diffusion_eff(lambda_d, lambda_a_list[idx])
        eff_list.append(eff)

    eff_list = np.asarray(eff_list)
    return eff_list, l_d_scaled, l_a_scaled

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

def loopThroughTrapEffs(ax, df, plot_name): 
    colors = ['red', 'blue', 'pink']
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        label_long = fr"$\tau_{{decay}}=${row["tau_decay"]*1e6:.2g} $\mathrm{{\mu}}s$"+"\n"+ \
            fr"$\tau_{{trap}}=${row["tau_trap"]*1e6:.2g} $\mathrm{{\mu}}s$"+"\n"+ \
            fr"D={row["d_diffuse"]} $\mathrm{{m^2}}$/s"
        ax.plot(row["t_abs_list"]*1e9, row["eff_list"]*100, marker='o', 
            label=label_long, 
            color=colors[idx])
    ax.set_xlabel(r'$t_{abs}$ (nm)')
    ax.set_ylabel('QP trapping efficiency (%)')
    ax.set_title(f'QP trapping ({plot_name})')
    ax.grid(True)
    ax.axvline(x=600, label=r'$t_{abs}$=600 $\mu$m', color='black')

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plotTrappEff(d_diffuse, tau_decay, tau_trap, t_abs_list, plot_dir, plot_name): 
    df_default = pd.DataFrame([{"d_diffuse": d_diffuse, "tau_decay": tau_decay, 
        "tau_trap": tau_trap, "t_abs_list": t_abs_list, "eff_list": 0}])

    vary_list = [0.1, 10]
    scaled_dfs = []
    for vary in vary_list:
        for var in ["d_diffuse", "tau_decay", "tau_trap"]:
            df_scaled = df_default.copy()
            df_scaled[var] = df_scaled[var] * vary
            scaled_dfs.append(df_scaled)

    df_scaled = pd.concat(scaled_dfs, ignore_index=True)
    df_all_effs = pd.concat([df_scaled, df_default], ignore_index=True)
    df_all_effs['eff_list'] = df_all_effs['eff_list'].astype(object)
    for idx, row in df_all_effs.iterrows():
        eff_list_row = getTrapeff(row["d_diffuse"], 
            row["tau_decay"], row["tau_trap"], row["t_abs_list"], plot_name)
        df_all_effs.at[idx, 'eff_list'] = eff_list_row

    df_diffuse = df_all_effs[(df_all_effs['tau_decay'] == tau_decay) 
        & (df_all_effs['tau_trap'] == tau_trap)]
    df_decay = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_trap'] == tau_trap)]
    df_trap = df_all_effs[(df_all_effs['d_diffuse'] == d_diffuse) 
        & (df_all_effs['tau_decay'] == tau_decay)]

    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    # Save figure
    os.makedirs(plot_dir, exist_ok=True)
    loopThroughTrapEffs(axes[0], df_diffuse, plot_name)
    loopThroughTrapEffs(axes[1], df_decay, plot_name)
    loopThroughTrapEffs(axes[2], df_trap, plot_name)

    plt.tight_layout()
    plt.savefig(plot_dir+f"/{plot_name}_trap_eff_all"+".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(plot_dir+f"/{plot_name}_trap_eff_all"+".png", dpi=300, bbox_inches='tight')
    plt.close()
    return True

def main():
    # plot_dir = "output/2025-5-12-testdiffusion"
    # # diffuse_constant_plot(plot_dir, data_sum_config.diffusion_constant_summary)

    # ld = 300*1e-6 # m
    # d_diffuse = 0.02 #m^2/s
    # tau_decay = ld**2/d_diffuse #s
    # la = 500*1e-6 # m
    # tau_trap = la**2/d_diffuse # m/s
    # # l_abs = 100*1e-6 # m
    # # l_abs = ld/1000 # m
    # l_abs_list = np.linspace(ld/1000, ld*2, 100)
    # plotDiffusionEff(d_diffuse, tau_decay, tau_trap, l_abs_list, plot_dir, 
    #     thickness_config=data_sum_config.scdms_film_thickness_lambdas)

    # # t_abs = 600*1e-9 # m
    # # t_abs_list = np.linspace(t_abs/60, t_abs*10, 100)
    # # # trap_eff = getTrapeff(tau_decay, tau_trap, d_diffuse, t_abs)
    # # plotTrappEff(d_diffuse, tau_decay, tau_trap, t_abs_list, plot_dir, "vertical")

    # # t_abs = 600*1e-9 # m
    # # t_abs_list = np.linspace(t_abs/60, t_abs*10, 100)
    # # # trap_eff = getTrapeff(tau_decay, tau_trap, d_diffuse, t_abs)
    # # plotTrappEff(d_diffuse, tau_decay, tau_trap, t_abs_list, plot_dir, "sideway")

# ##############################################
# ##############################################
# ##############################################
#     plot_dir = "output/2025-7-3-diffusion-eff/"
#     ld = 300*1e-6 # m
#     d_diffuse = 0.02 #m^2/s
#     tau_decay = ld**2/d_diffuse #s
#     la = 500*1e-6 # m
#     tau_trap = la**2/d_diffuse # m/s
#     print(tau_decay, tau_trap)
#     l_abs_list = np.linspace(ld/1000, ld*2, 100)

#     title = 'check_eff_changes_overlap_w'
#     plotDiffusionEff(d_diffuse, tau_decay, tau_trap, l_abs_list, plot_dir+title)

##############################################
##############################################
##############################################
    plot_dir = "output/2025-7-7-diffusion-eff-second/"
    # ld = 300*1e-6 # m
    # d_diffuse = 0.02 #m^2/s
    # tau_decay = ld**2/d_diffuse #s
    # la = 500*1e-6 # m
    # tau_trap = la**2/d_diffuse # m/s
    # print(tau_decay, tau_trap)
    # l_abs_list = np.linspace(ld/1000, ld*2, 100)

    # title = 'check_eff_changes_overlap_l'
    # plotDiffusionEff(d_diffuse, tau_decay, tau_trap, l_abs_list, plot_dir+title)

    ld = 50*1e-6 # m
    d_diffuse = 0.02 #m^2/s
    d_diffuse_scaled = d_diffuse/6
    tau_decay = ld**2/d_diffuse_scaled #s
    la = 13.9*1e-6 # m
    tau_trap = la**2/d_diffuse_scaled # m/s
    print(d_diffuse_scaled, tau_decay, tau_trap)
    l_abs_list = np.linspace(ld/1000, ld*2, 100)

    # title = 'check_second_diffusion'
    title = 'check_second_diffusion_5mum'
    plotDiffusionEff(d_diffuse_scaled, tau_decay, tau_trap, l_abs_list, plot_dir+title,
        w_abs=1*1e-6, # m
        )

if __name__ == "__main__":
    main()
