
import os
import numpy as np 

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from time import time

import argparse

cuda = torch.cuda.is_available()


# Command line config
parser = argparse.ArgumentParser(description='AE Batch Runner')

parser.add_argument('--start-point', type=int, default=0, metavar='N',
                    help='start point for batch run (default: 0)')

parser.add_argument('--end-point', type=int, default=1000, metavar='N',
                    help='end point for batch run (default: 1000)')

args = parser.parse_args()

RAW_DATA_DIR = "../Data/Raw/"
PROCESSED_DATA_DIR = "../Data/Processed/"

class CovariateDataset(Dataset):
    def __init__(self, file_name_pattern, file_name_args):
        self.file_name = file_name_pattern.format(*file_name_args)
        self.data = np.loadtxt(RAW_DATA_DIR + self.file_name + ".csv", delimiter=",")[:, 1:] # remove bias
        
    def __getitem__(self, index):
        return (self.data[index].astype(float), 0)

    def __len__(self):
        return self.data.shape[0]
    
    def save_processed_data(self, data, loss):
        name = PROCESSED_DATA_DIR + self.file_name+"_{}.csv".format(loss)
        np.savetxt(name, data, delimiter=",")


# ### Define Model

# In[4]:


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(True),
            nn.Linear(128, 4))
        self.decoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(True),
            nn.Linear(128, 10))

    def forward(self, x):
        encoded_values = self.encoder(x)
        x = self.decoder(encoded_values)
        return x, encoded_values


# ### Train and Process Utils

# In[5]:


def reconstruction_sparsity_loss(output, target, encoded_values):
        sparsity_scalar = Variable(torch.FloatTensor([0.0005]))
        if cuda:
            sparsity_scalar = sparsity_scalar.cuda()
            
        mse_loss = nn.MSELoss()
        reconstruction_loss = mse_loss(output, target)
        sparsity_loss = encoded_values.abs().sum()*sparsity_scalar
        return reconstruction_loss + sparsity_loss 
    
def reconstruction_loss(output, target, encoded_values):
    mse_loss = nn.MSELoss()
    reconstruction_loss = mse_loss(output, target)
    return reconstruction_loss


# In[39]:


loss_functions = ["reconstruction", "sparsity"]

def train_model(model_class, dataset, dataset_number, loss="reconstruction", verbose=True):
    model = model_class()
    if cuda:
        model = model.cuda()

    num_epochs = 10000
    batch_size = 1000
    learning_rate = 1e-1
    lr_sched = True
    
    
    if loss == loss_functions[0]:
        criterion = reconstruction_loss
    elif loss == loss_functions[1]:
        criterion = reconstruction_sparsity_loss
         
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2000, 5000], gamma=0.1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    final_loss = None
    
    for epoch in range(num_epochs):
        if lr_sched:
            scheduler.step()

        for data in dataloader:
            data_batch, _ = data
            data_batch = Variable(data_batch)
            data_batch = data_batch.float()

            if cuda:
                data_batch = data_batch.cuda()

            # Forward pass
            output, encoded_values = model(data_batch)

            loss = criterion(output, data_batch, encoded_values)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch%int(num_epochs/10) == int(num_epochs/10)-1 and verbose:
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data[0]))
        
        if epoch == (num_epochs-1):
            final_loss = loss.data[0]
            print("Final loss: loss:{:.4f}".format(final_loss))

    torch.save(model.state_dict(), "../Models/simple_autoencoder_{}.pth".format(dataset_number))
    return model, final_loss

def encode_data(model, dataset, loss):
    all_data = torch.from_numpy(dataset.data)
    all_data = Variable(all_data)
    all_data = all_data.float()
    
    if cuda:
        all_data = all_data.cuda()

    output = model.encoder(all_data)
    
    if cuda:
        output = output.cpu()
        
    dataset.save_processed_data(output.data.numpy(), loss)
    return output.data.numpy()


# In[7]:


# dataset = CovariateDataset("n_{}_model_{}_v_{}_covar_data", [1000, "A_add_lin", 1])
# trained_model, final_loss = train_model(
#                                     autoencoder,
#                                     dataset,
#                                     loss="sparsity",
#                                     verbose=True)
# encode_data(trained_model, dataset, "sparsity")


# ### Train and Encode

# In[40]:


def run_for_range(start, end):
    models_to_rerun = []
    datasets_to_process = range(start, end)
    assignment_model_names = ['A_add_lin', 'B_add_mild_nlin', 'C_add_mod_nlin', 'D_mild_nadd_lin',
                         'E_mild_nadd_mild_nlin', 'F_mod_nadd_lin', 'G_mod_nadd_mod_nlin']

    for dataset_number in datasets_to_process:
        print("Starting run for Dataset {}".format(dataset_number))

        for model_name in assignment_model_names:
            print("-- Running for model name: ", model_name)

            for loss_type in loss_functions:
                print("---- Running for loss: ", loss_type)

                start = time()

                dataset = CovariateDataset("n_{}_model_{}_v_{}_covar_data", [1000, model_name, dataset_number])
                trained_model, final_loss = train_model(
                                                    autoencoder,
                                                    dataset,
                                                    dataset_number,
                                                    loss=loss_type,
                                                    verbose=True)
                encode_data(trained_model, dataset, loss=loss_type)

                print("---- Done in ", time() - start, " seconds\n")

                # Catch bad runs
                if loss_type == loss_functions[0] and final_loss > 0.30:
                    models_to_rerun.append((model_name, dataset_number, loss_type))
                elif loss_type == loss_functions[1] and final_loss > 1.0:
                    models_to_rerun.append((model_name, dataset_number, loss_type))

        print("================\n\n")

    print("Rerun: ", models_to_rerun)
    return models_to_rerun


# In[38]:


run_for_range(args.start_point, args.end_point)


# In[8]:


# models_to_rerun = [('A_add_lin', 12, 'sparsity'), ('G_mod_nadd_mod_nlin', 40, 'sparsity')]

for model_name, dataset_number, loss_type in models_to_rerun:
    dataset = CovariateDataset("n_{}_model_{}_v_{}_covar_data", [1000, model_name, dataset_number])
    trained_model, final_loss = train_model(
                                        autoencoder,
                                        dataset,
                                        loss=loss_type,
                                        verbose=True)
    encode_data(trained_model, dataset, loss=loss_type)

