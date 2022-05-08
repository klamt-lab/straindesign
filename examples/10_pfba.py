from cobra.test import create_test_model
import straindesign as sd
model = create_test_model('textbook')
sol0 = sd.fba(model,pfba=0)
sol1 = sd.fba(model,pfba=1)
sol2 = sd.fba(model,pfba=2)
print('none')