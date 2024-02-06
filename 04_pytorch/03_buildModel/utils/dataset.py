import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, root='./datasets', transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        if mode == 'train':
            self.dataset_path = os.path.join(self.root, 'train')
        elif mode == 'val':
            self.dataset_path = os.path.join(self.root, 'test')
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented yet..")
        
        # Load image paths and corresponding label
        self.image_paths = []
        self.labels = []
        
        # Get image paths and labels
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))
        
        # Get class names
        self.class_names = list(set(self.labels))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

thisone = CustomDataset()
    
    