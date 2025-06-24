diffusion_constant_summary = {
"hsieh_1968": {
    "diffusion_constant": 22.5*1e-4,
    "unit": "m**2/s",
    "source": "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.20.1502",
    "temp_K": 0.38,
    "material": "al", 
    "film_thickness_m": 300*1e-10,
    "method": "measured from diffusion length and tau_r",
},
"hsieh_1968_cal": {
    "diffusion_constant": 23.6*1e-4,
    "unit": "m**2/s",
    "source": "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.20.1502",
    "temp_K": 0.38,
    "material": "al", 
    "film_thickness_m": 300*1e-10,
    "method": "calculated from mfp (measured by conductivity) and group velocity",
},
"scdms_jeff_300": {
    "diffusion_constant": 0.010,
    "unit": "m**2/s",
    "source": "https://www.slac.stanford.edu/exp/cdms/ScienceResults/Theses/yen.pdf",
    "temp_K": 0.035,
    "material": "al", 
    "film_thickness_m": 300*1e-9,
    "method": "calculated from mfp (measured by RRR) and scaled to tc",
},
"scdms_jeff_600": {
    "diffusion_constant": 0.021,
    "unit": "m**2/s",
    "source": "https://www.slac.stanford.edu/exp/cdms/ScienceResults/Theses/yen.pdf",
    "temp_K": 0.035,
    "material": "al", 
    "film_thickness_m": 600*1e-9,
    "method": "calculated from mfp (measured by RRR) and scaled to tc",
},
"dave_almn": {
    "diffusion_constant": 19.8*1e-4,
    "diffusion_constant_err": 5.3*1e-4,
    "unit": "m**2/s",
    "source": "https://pubs.aip.org/aip/apl/article/100/23/232601/282121/Position-and-energy-resolved-particle-detection",
    "temp_K": "NA",
    "material": "almn", 
    "film_thickness_m": "NA",
    "method": "fitted from banana experiment with tau",
},
# "dave_ta": {
#     "diffusion_constant": 13.5*1e-4,
#     "diffusion_constant_err": 1.8*1e-4,
#     "unit": "m**2/s",
#     "source": "https://pubs.aip.org/aip/apl/article/100/23/232601/282121/Position-and-energy-resolved-particle-detection",
#     "temp_K": "NA",
#     "material": "ta", 
#     "film_thickness_m": 600*1e-9,
#     "method": "fitted from banana experiment with tau",
# },
"crest_al": {
    "diffusion_constant": 2.5*1e-4,
    "unit": "m**2/s",
    "source": "https://www.sciencedirect.com/science/article/pii/S0168900201006210",
    "temp_K": "NA",
    "material": "al", 
    "film_thickness_m": 1000*1e-6,
    "method": "fitted from banana experiment with tau",
},
"peter_al_mag": {
    "diffusion_constant": 33.8*1e-4,
    "unit": "m**2/s",
    "source": "Karthik's local file",
    "temp_K": 1,
    "material": "al", 
    "film_thickness_m": 20*1e-9,
    "method": "other ways from magnetic response?",
},
}

scdms_film_thickness_lambdas = {
"900*1e-9": { #m
    "la_m": 1250*1e-6, # m
    "ld_m": 500*1e-6, # m
},
"300*1e-9": {
    "la_m": 200*1e-6,
    "ld_m": 200*1e-6,
}}
