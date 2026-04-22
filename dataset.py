import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.JPEG', '.JPG']


def is_image_file(filename):
    """
    检查文件是否为图像文件
    
    Args:
        filename (str): 待检查的文件名
        
    Returns:
        bool: 如果文件扩展名在支持的图像格式列表中则返回True，否则返回False
    """
    ext = filename.lower().split('.')[-1] #lower小写，split分割字符串，[-1]取最后一个元素，即扩展名
    return ext in IMG_EXTENSIONS


def get_files(data_dir):
    img_list = []
    print(f"[DEBUG] Searching in: {data_dir}")
    
    # 标准化路径（处理 Windows 反斜杠）
    data_dir = os.path.normpath(data_dir)
    
    for root, dirs, files in os.walk(data_dir):
        print(f"[DEBUG] Walking: {root}, files: {len(files)}")
        for img in files:
            ext = img.lower().split('.')[-1]
            if ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm']:
                img_path = os.path.join(root, img)
                img_list.append(img_path)
    img_list = sorted(img_list)
    print(f"[DEBUG] Found {len(img_list)} images")
    return img_list


# steganography dataset
class StegDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.img_list = get_files(data_dir)
        self.img_num = len(self.img_list)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        img = img.convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return self.img_num


def get_data_transform(use_aug, h, w):
    if use_aug:#数据增强
        transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    return transform


def steg_data_loader(mode, opt):
    if mode == 'train':
        data_dir = os.path.join(opt.data_dir, 'train')
        use_aug = opt.use_aug
        shuffle = True
    else:
        data_dir = os.path.join(opt.data_dir, 'val') if mode == 'val' else opt.data_dir
        use_aug = False
        shuffle = False

    transform = get_data_transform(use_aug, opt.image_height, opt.image_width)

    dataset = StegDataset(data_dir, transform)
    assert dataset

    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, 
                             num_workers=opt.workers, pin_memory=True, drop_last=False)

    return data_loader
