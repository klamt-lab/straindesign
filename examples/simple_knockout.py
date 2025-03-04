﻿# Example of knockout optimization\n\n'''\nThis example demonstrates how to use StrainDesign for knockout optimization\n'''\n\nimport straindesign as sd\n\n# Load a model\nmodel_path = 'path/to/model.xml'\n\n# Run knockout optimization\nresult = sd.optimize_knockouts(model_path, objective='BIOMASS')\nprint(result.interventions)