from cobra.io import load_model, read_sbml_model
from cobra import Configuration
import straindesign as sd
from straindesign.names import *
from numpy import inf, nan
import pandas as pd

cobra_conf = Configuration()
cobra_conf.solver = 'cplex'
models = [
    'e_coli_core', 'straindesign/examples/ECC2.sbml', 'iJO1366', 'iML1515',
    'iJN746', 'iJN1463', 'iMM904', 'straindesign/examples/yeast-GEM.xml',
    'iMM1415', 'iCHOv1', 'Recon3D'
]
comp_no = [
    'number of model reactions',
    '1 after removing blocked',
    '2 after EFM compression',
    '3 after parallel compression',
    '3 after EFM compression',
    '4 after parallel compression',
    '5 after EFM compression',
    '6 after parallel compression',
    '7 after EFM compression',
    '8 after parallel compression',
    '9 after EFM compression',
    '10 after parallel compression',
    '11 after EFM compression',
    '12 after parallel compression',
    '13 after EFM compression',
    '14 after parallel compression',
]
compr = pd.DataFrame(index=comp_no, columns=models)

for i, model_name in enumerate(models):
    print('loading ' + model_name)
    if 'yeast-GEM' in model_name or 'ECC2' in model_name:
        model = read_sbml_model(model_name)
    else:
        model = load_model(model_name)

    bound_thres = 1000
    if any([
            any([abs(b) >= bound_thres
                 for b in r.bounds])
            for r in model.reactions
    ]):
        print('  Removing reaction bounds when larger than the threshold of ' +
              str(bound_thres) + '.')
        for i in range(len(model.reactions)):
            if model.reactions[i].lower_bound <= -bound_thres:
                model.reactions[i].lower_bound = -inf
            if model.reactions[i].upper_bound >= bound_thres:
                model.reactions[i].upper_bound = inf

    # FVAs to identify blocked, irreversible and essential reactions, as well as non-bounding bounds
    print('  FVA to identify blocked reactions and irreversibilities.')
    flux_limits = sd.fva(model)
    if cobra_conf.solver in ['scip', 'glpk']:
        tol = 1e-10  # use tolerance for tightening problem bounds
    else:
        tol = 0.0
    for (reac_id, limits) in flux_limits.iterrows():
        r = model.reactions.get_by_id(reac_id)
        # modify _lower_bound and _upper_bound to make changes permanent
        if r.lower_bound < 0.0 and limits.minimum - tol > r.lower_bound:
            r._lower_bound = -inf
        if limits.minimum >= tol:
            r._lower_bound = max([0.0, r._lower_bound])
        if r.upper_bound > 0.0 and limits.maximum + tol < r.upper_bound:
            r._upper_bound = inf
        if limits.maximum <= -tol:
            r._upper_bound = min([0.0, r._upper_bound])

    print('Compressing Network (' + str(len(model.reactions)) + ' reactions).')
    compr.loc[comp_no[0], model_name] = len(model.reactions)
    # compress network by lumping sequential and parallel reactions alternatingly.
    # Remove conservation relations.
    blocked_reactions = sd.remove_blocked_reactions(model)
    print('  Removed blocked reactions (' + str(len(model.reactions)) +
          ' reactions).')
    compr.loc[comp_no[1], model_name] = len(model.reactions)
    print('  Translating stoichiometric coefficients to rationals.')
    sd.stoichmat_coeff2rational(model)
    print('  Removing conservation relations.')
    sd.remove_conservation_relations(model)
    odd = True
    run = 1
    while True:
        # np.savetxt('Table.csv',create_stoichiometric_matrix(cmp_model),'%i',',')
        if odd:
            print('  Compression ' + str(run) +
                  ': Applying compression from EFM-tool module.')
            subT, reac_map_exp = sd.compress_model(model)
        else:
            print('  Compression ' + str(run) + ': Lumping parallel reactions.')
            subT, reac_map_exp = sd.compress_model_parallel(model)
        sd.remove_conservation_relations(model)
        if subT.shape[0] > subT.shape[1]:
            print('  Reduced to ' + str(subT.shape[1]) + ' reactions.')
            compr.loc[comp_no[run + 1], model_name] = len(model.reactions)
            run += 1
            odd = not odd
        else:
            print('  Last step could not reduce size further (' +
                  str(subT.shape[0]) + ' reactions).')
            print('  Network compression completed. (' + str(run - 1) +
                  ' compression iterations)')
            print('  Translating stoichiometric coefficients back to float.')
            sd.stoichmat_coeff2float(model)
            break
compr.to_csv('Compression.csv')
