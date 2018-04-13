import numpy as np
assignment_model_names = ['A_add_lin', 'B_add_mild_nlin', 'C_add_mod_nlin', 'D_mild_nadd_lin',
                     'E_mild_nadd_mild_nlin', 'F_mod_nadd_lin', 'G_mod_nadd_mod_nlin']
rerun = []
for model in assignment_model_names:
  for i in range(50):
    data = np.loadtxt("Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i), delimiter=",")
    if np.array_equal(data[:500, 0], data[500:, 0]) and data[0,3] == data[999, 3]:
      print(model, i, "Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i))
      rerun.append((model, i, "sparsity"))

print(str(rerun))
