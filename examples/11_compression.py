from cobra.io import load_model
from cobra import Configuration
import straindesign as sd
from straindesign.names import *
from numpy import inf
cobra_conf = Configuration()
cobra_conf.solver = 'cplex'
model = load_model('iJO1366')

bound_thres = 1000
if any([any([abs(b)>=bound_thres for b in r.bounds]) for r in model.reactions]):
    print('  Removing reaction bounds when larger than the threshold of '+str(bound_thres)+'.')
    for i in range(len(model.reactions)):
        if model.reactions[i].lower_bound <= -bound_thres:
            model.reactions[i].lower_bound = -inf
        if model.reactions[i].upper_bound >=  bound_thres:
            model.reactions[i].upper_bound =  inf

print('Compressing Network ('+str(len(model.reactions))+' reactions).')
# compress network by lumping sequential and parallel reactions alternatingly. 
# Remove conservation relations.
print('  Removing blocked reactions ('+str(len(model.reactions))+' reactions).')
blocked_reactions = sd.remove_blocked_reactions(model)
print('  Translating stoichiometric coefficients to rationals.')
sd.stoichmat_coeff2rational(model)
print('  Removing conservation relations.')
sd.remove_conservation_relations(model)
odd = True
run = 1
while True:
    # np.savetxt('Table.csv',create_stoichiometric_matrix(cmp_model),'%i',',')
    if odd:
        print('  Compression '+str(run)+': Applying compression from EFM-tool module.')
        subT, reac_map_exp = sd.compress_model(model)                  
    else:
        print('  Compression '+str(run)+': Lumping parallel reactions.')
        subT, reac_map_exp = sd.compress_model_parallel(model)
    sd.remove_conservation_relations(model)
    if subT.shape[0] > subT.shape[1]:
        print('  Reduced to '+str(subT.shape[1])+' reactions.')
        run += 1
        odd = not odd 
    else:
        print('  Last step could not reduce size further ('+str(subT.shape[0])+' reactions).')
        print('  Network compression completed. ('+str(run-1)+' compression iterations)')
        print('  Translating stoichiometric coefficients back to float.')
        sd.stoichmat_coeff2float(model)
        break