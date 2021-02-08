class PermEstimateDataset(Dataset):
  
    def __init__(self, ext='data'):
    
        data = h5py.File(ext+'_128_20um_11_140_2channels.mat')
        self.field = np.transpose(data['train_Field'])
        self.perm = np.transpose(data['train_perm'])
        
    def __len__(self):
        
        return len(self.perm)
  
    def __getitem__(self, idx):
 
        field = self.field[idx]
        perm = self.perm[idx]
        """Convert ndarrays to Tensors."""
        return {'field': torch.from_numpy(field).float(),
                'perm': torch.from_numpy(perm).float()
               }