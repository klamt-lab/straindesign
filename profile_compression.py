"""Profile compression to find bottlenecks."""
import time
import cProfile
import pstats
import io
import cobra
from straindesign.compression import compress_cobra_model

print('Loading iML1515...')
model = cobra.io.load_model('iML1515')
print(f'Model: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites')

# Profile the compression
print('\nProfiling compression...')
profiler = cProfile.Profile()
profiler.enable()

start = time.time()
result = compress_cobra_model(model)
elapsed = time.time() - start

profiler.disable()

print(f'\nTotal time: {elapsed:.2f}s')
print(f'Compressed: {len(result.compressed_model.reactions)} reactions')
print(f'Objective: {result.compressed_model.optimize().objective_value:.4f}')

# Print top 30 time consumers
print('\n=== Top 30 Time Consumers ===')
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
