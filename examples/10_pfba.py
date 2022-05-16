from cobra.io import load_model
from cobra import Configuration
import straindesign as sd
cobra_conf = Configuration()
cobra_conf.solver = 'cplex'
model = load_model('e_coli_core')
# sol0 = sd.fba(model,pfba=0)
# sol1 = sd.fba(model,pfba=1)
# sol2 = sd.fba(model,pfba=2)
# print('none')

r1 = ('-EX_o2_e')
r2 = ('BIOMASS_Ecoli_core_w_GAM')
y2 = ('BIOMASS_Ecoli_core_w_GAM','-EX_glc__D_e')
y3 = ('EX_etoh_e','-EX_glc__D_e')
# plot2 = sd.plot_flux_space(model,(y1,y2),constraints='EX_o2_e >= -1',points=10)
plot1 = sd.plot_flux_space(model,(r2,y2),constraints='EX_o2_e >= -25',points=15)
# plot2 = sd.plot_flux_space(model,(r2,y2,r1),constraints='EX_o2_e >= -25',points=15)
