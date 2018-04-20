import numpy as np 

treated = np.loadtxt("./Raw/nswre74_treated.txt", delimiter="  ", dtype=str)[:, 1:].astype(float)
control = np.loadtxt("./Raw/nswre74_control.txt", delimiter="  ", dtype=str)[:, 1:].astype(float)

data = np.vstack([treated, control])
assignments = data[:, 0]
outcomes = data[:, 9]
covars = data[:, 1:9]

print("Exp treat effect:", np.mean(outcomes[assignments==1]) - np.mean(outcomes[assignments==0]))

np.savetxt("./Raw/nsw74_all_assignments.csv", assignments, delimiter=",")
np.savetxt("./Raw/nsw74_all_outcomes.csv", outcomes, delimiter=",")
np.savetxt("./Raw/nsw74_all_covars.csv", covars, delimiter=",")