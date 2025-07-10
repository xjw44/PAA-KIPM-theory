########################################## nominal config 
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

# param_dict_abs = {"d_diffuse": d_diffuse, 
# "tau_trap": tau_trap,
# "tau_decay": tau_decay
# }
# param_dict_over = {"d_diffuse_over": d_diffuse_over, 
# "tau_trap_over": tau_trap_over,
# "tau_decay_over": tau_decay_over
# }

# ########################################## larger tau_trap_over
# tau_trap_over = 0.188*1e-6

# param_dict_abs = {"d_diffuse": d_diffuse, 
# "tau_trap": tau_trap,
# "tau_decay": tau_decay
# }
# param_dict_over = {"d_diffuse_over": d_diffuse_over, 
# "tau_trap_over": tau_trap_over,
# "tau_decay_over": tau_decay_over
# }

########################################## larger tau_decay
tau_trap_over = 0.188*1e-6
tau_decay = ld**2/d_diffuse*10 #s
tau_decay_over = ld_over**2/d_diffuse_over*10 #s

param_dict_abs = {"d_diffuse": d_diffuse, 
"tau_trap": tau_trap,
"tau_decay": tau_decay
}
param_dict_over = {"d_diffuse_over": d_diffuse_over, 
"tau_trap_over": tau_trap_over,
"tau_decay_over": tau_decay_over
}






