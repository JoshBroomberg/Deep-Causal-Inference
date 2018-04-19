import pickle
import csv

results_prefix = "./Results/"
assignment_model_names = ['A_add_lin', 'B_add_mild_nlin', 'C_add_mod_nlin', 'D_mild_nadd_lin',
                     'E_mild_nadd_mild_nlin', 'F_mod_nadd_lin', 'G_mod_nadd_mod_nlin']

results_files = {
    "Logistic Propensity Matching": "Original/est_logistic_prop_matching_est_runs_1000_n_1000",
    
    # GenMatch
    "GenMatch": "Original/est_genmatch_est_runs_1000_n_1000",
    
    # AE
    "GenMatch - Reconstruction AE (yes prop scores - compressed, eval on compresses)": "AE/Reconstruction/est_genmatch_est_runs_200_n_1000",
    "GenMatch - Reconstruction AE (no prop scores, eval on compresses)": "AE/Reconstruction/nopropscores/est_genmatch_est_runs_200_n_1000",
    "GenMatch - Reconstruction AE (no prop scores, eval on original)": "AE/Reconstruction/evalonoriginal/est_genmatch_est_runs_200_n_1000",
    "GenMatch - Reconstruction AE (yes prop scores - uncompressed, eval on original)": "AE/Reconstruction/evalonoriginal_withp/est_genmatch_est_runs_200_n_1000",
    "GenMatch - Reconstruction AE (yes prop scores - uncompressed, eval on compressed)": "AE/Reconstruction/withp/est_genmatch_est_runs_200_n_1000",
    "GenMatch - Sparse AE (no prop scores, eval on compresses)": "AE/Sparsity/est_genmatch_est_runs_50_n_1000",
    "GenMatch - Sparse AE (yes prop scores, eval on compresses)": "AE/Sparsity/withp/est_genmatch_est_runs_50_n_1000",
    "GenMatch - Sparse AE (yes prop scores, eval on original)": "AE/Sparsity/evalonoriginal_withp/est_genmatch_est_runs_50_n_1000",
    
    # VAE
    "GenMatch - VAE (no prop score, eval on compressed)": "VAE/est_genmatch_est_runs_200_n_1000",
    "GenMatch - VAE (yes prop score, eval on compressed)": "VAE/withp/est_genmatch_est_runs_200_n_1000",
    "GenMatch - VAE (yes prop score, eval on original)": "VAE/evalonoriginal_withp/est_genmatch_est_runs_200_n_1000",
    "Mahalanobis - VAE (no prop score)": "VAE/md/est_mahalanobis_matching_runs_200_n_1000",
    "Mahalanobis - VAE (yes prop score)": "VAE/md_withp/est_mahalanobis_matching_runs_200_n_1000",
    "Bhattacharyya Distance - VAE (no prop score)": "VAE/Z2/bhat/est_mahalanobis_matching_runs_200_n_1000",
    
    # NN Prop
    "Neural Propensity Matching": "NN/est_logistic_prop_matching_est_runs_200_n_1000",
    "GenMatch - NN Prop Score": "NN/genmatch/est_genmatch_est_runs_250_n_1000",
    # "GenMatch - NN Prop Score + Logistic Prop Score": "NN/genmatch_bothp",
    
    # GENMATCH AE
    "GenMatch - NN Prop Score + AE Reconstruction (eval on compressed)": "NN/genmatch_reconstruction/est_genmatch_est_runs_200_n_1000"
    # "GenMatch - NN Prop Score + AE Reconstruction (eval on original)": "NN/genmatch_reconstruction_evalonoriginal/est_genmatch_est_runs_200_n_1000"
    
    # GENMATCH VAE
    # "GenMatch - NN Prop Score + VAE (eval on compressed)": "NN/genmatch_vae/est_genmatch_est_runs_200_n_1000"
    # "GenMatch - NN Prop Score + VAE (eval on compressed)": "NN/genmatch_vae_evaloriginal/est_genmatch_est_runs_200_n_1000"

}
print(str("\t".join(assignment_model_names)))

with open('results.csv', 'w') as csvfile:
    
    result_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for metric in ["Bias", "RMSE"]:
        for result_name, result_file in results_files.items():
            result_dict = pickle.load(open(results_prefix + result_file + ".p", "rb"))
            metric_values = [str(result_dict[model][metric]) for model in assignment_model_names]
            result_writer.writerow([result_name] + metric_values)


