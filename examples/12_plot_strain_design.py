from cobra.io import load_model
from cobra import Configuration
import straindesign as sd
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

cobra_conf = Configuration()
# cobra_conf.solver = 'cplex'
model = load_model('e_coli_core')
# sol0 = sd.fba(model,pfba=0)
# sol1 = sd.fba(model,pfba=1)
# sol2 = sd.fba(model,pfba=2)
# print('none')

r1 = ('-EX_o2_e')
r2 = ('BIOMASS_Ecoli_core_w_GAM')
y2 = ('BIOMASS_Ecoli_core_w_GAM', '-EX_glc__D_e')
y3 = ('EX_etoh_e', '-EX_glc__D_e')
constraints = ['NADTRHD = 0',
               'NADH16 = 0',
               'LDH_D = 0',
               'PPC = 0']
# plot2 = sd.plot_flux_space(model,(y1,y2),constraints='EX_o2_e >= -1',points=10)
# plot1 = sd.plot_flux_space(model,(r2,y2),constraints='EX_o2_e >= -25',points=15)
# plot2 = sd.plot_flux_space(model,(r2,y2,r1),constraints='EX_o2_e >= -25',points=15)
# plot1 = sd.plot_flux_space(model,(r1,y2,y3),constraints='EX_o2_e >= -8',points=120)

_,_,plot2 = sd.plot_flux_space(model, (r1, r2, y3),
                           constraints=constraints,
                           points=30,
                           plt_backend='AGG',
                           show=False)

# plt.show()

def animate(angle):
    plot2._axes.view_init(20, angle)
    return plot2

plot2.figure.set_figheight(2.4)
plot2.figure.set_figwidth(2.6)
plot2._axes.tick_params(axis='both', which='major', labelsize=6, pad=0)
plot2._axes.tick_params(axis='both', which='minor', labelsize=6, pad=0)
plot2._axes.xaxis.set_tick_params(labelsize=6, pad=-2)
plot2._axes.yaxis.set_tick_params(labelsize=6, pad=-2)
plot2._axes.zaxis.set_tick_params(labelsize=6, pad=-2)
plot2._axes.xaxis.label.set_size(5)
plot2._axes.xaxis.labelpad = -7
plot2._axes.yaxis.label.set_size(5)
plot2._axes.yaxis.labelpad = -7
plot2._axes.zaxis.label.set_size(5)
plot2._axes.zaxis.labelpad = -7

# plt.show()

print('plotting animation')

ani = animation.FuncAnimation(
    plot2.figure, animate, save_count=360)

print('writing to file')

writer = animation.FFMpegWriter(
    fps=25, bitrate=1000)
ani.save("movie.gif", writer=writer)

