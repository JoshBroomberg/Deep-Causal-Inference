import numpy as np
assignment_model_names = ['A_add_lin', 'B_add_mild_nlin', 'C_add_mod_nlin', 'D_mild_nadd_lin',
                     'E_mild_nadd_mild_nlin', 'F_mod_nadd_lin', 'G_mod_nadd_mod_nlin']

rerun = [('A_add_lin', 169, 'sparsity'),
 ('D_mild_nadd_lin', 169, 'reconstruction'),
 ('E_mild_nadd_mild_nlin', 170, 'reconstruction'),
 ('E_mild_nadd_mild_nlin', 144, 'reconstruction'),
 ('F_mod_nadd_lin', 144, 'reconstruction'),
 ('C_add_mod_nlin', 146, 'reconstruction'),
 ('E_mild_nadd_mild_nlin', 147, 'reconstruction'),
 ('B_add_mild_nlin', 148, 'reconstruction'),
 ('B_add_mild_nlin', 149, 'reconstruction'),
 ('C_add_mod_nlin', 120, 'sparsity'),
 ('C_add_mod_nlin', 122, 'reconstruction'),
 ('D_mild_nadd_lin', 123, 'reconstruction')]

# for model in assignment_model_names:
#     for i in range(50, 200):
#         data = np.loadtxt("Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i),             delimiter=",")
#         if data[0, 0] == data[999, 0] and data[0,3] == data[999, 3]:
#             print(model, i, "Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i))
#             if (model, i, "sparsity") not in rerun:
#                 rerun.append((model, i, "sparsity"))

# print(str(rerun))

for model in assignment_model_names:

  for i in range(50):
    data = np.loadtxt("Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i), delimiter=",")
    if np.array_equal(data[:500, 0], data[500:, 0]) and data[0,3] == data[999, 3]:
      print(model, i, "Data/Processed/n_1000_model_{}_v_{}_covar_data_sparsity.csv".format(model, i))
      rerun.append((model, i, "sparsity"))

    for i in range(50, 200):
        data = np.loadtxt("Data/Processed/n_1000_model_{}_v_{}_covar_data_reconstruction.csv".format(model, i),             delimiter=",")
        if data[0, 0] == data[999, 0] and data[0,3] == data[999, 3]:
            print(model, i, "Data/Processed/n_1000_model_{}_v_{}_covar_data_reconstruction.csv".format(model, i))
            if (model, i, "reconstruction") not in rerun:
                rerun.append((model, i, "reconstruction"))

print(str(rerun))
