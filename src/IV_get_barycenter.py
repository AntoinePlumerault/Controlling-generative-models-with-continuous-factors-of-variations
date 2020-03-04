
import os
import cv2
import torch
import numpy as np

from torch.utils import data
from torch.autograd import Variable
from absl import app, flags, logging

from utils.save_images import save_images
from models.saliency.resnet import build_model

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size_IV', 1, "")

def compute_barycenter(image):
    X = np.linspace(-0.5, 0.5, image.shape[0])
    Y = np.linspace(-0.5, 0.5, image.shape[1])
    coord = np.meshgrid(X, Y)
    weights = coord * image
    x, y = np.sum(weights, axis=(1, 2)) / np.sum(image)
    return x, y, np.mean(image)

def get_barycenter():

    experiment_dir = os.path.join(
        FLAGS.output_dir, 
        '{}'.format(FLAGS.transform), 
        '{}'.format(FLAGS.y))

    os.makedirs(os.path.join(
        experiment_dir, 'barycenters', 'saliency_maps'), exist_ok=True)
   
# =============================== LOAD MODEL ===================================

    model = build_model()
    model.load_state_dict(torch.load(os.path.join(
        'models', 'saliency', 'final.pth')))
    model.eval().cuda()

# =============================== LOAD DATA ====================================

    class ImageDataTest(data.Dataset):
        def __init__(self):
            self.data_root = os.path.join(experiment_dir, 'latent_traversals')
            self.image_list = sorted(os.listdir(self.data_root))
            self.image_num = len(self.image_list)

        def __getitem__(self, item):
            path = os.path.join(self.data_root, self.image_list[item])
            image = cv2.imread(path)
            image = np.array(image, dtype=np.float32) 
            image = image - np.array((104.00699, 116.66877, 122.67892))
            image = image.transpose((2,0,1))
            image = torch.Tensor(image)
            return {
                'image': image, 
                'name': self.image_list[item % self.image_num], 
            }

        def __len__(self):
            return self.image_num

    data_loader = data.DataLoader(
        dataset=ImageDataTest(), batch_size=FLAGS.batch_size_IV, shuffle=False, 
        num_workers=8, pin_memory=False)

# ================================ RUN MODEL ===================================
    
    filename = os.path.join(experiment_dir, 'barycenters',  'barycenters.csv')
    with open(filename, 'w') as f:
        f.write('name,x,y,mean\n')
        for i, batch in enumerate(data_loader):
            images = batch['image']
            name = batch['name'][0] 

            with torch.no_grad():
                images = Variable(images)
                images = images.cuda()
                preds = model(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                
                # Compute barycenter:
                x, y, mean = compute_barycenter(pred)
                template = '{},{:4.3f},{:4.3f},{:4.3f}\n'
                f.write(template.format(name, x, y, mean))
                
                pred = np.array([pred[:,:]])
                filename = os.path.join(
                    experiment_dir, 'barycenters', 'saliency_maps', name)
                save_images(pred, filename)          


if __name__ == '__main__':
    # output directory
    flags.DEFINE_string('output_dir', os.path.join('..', 'outputs', 'test'), 
        "output directory")

    # choice of the model
    flags.DEFINE_enum('image_size', '128', ['128', '256', '512'], 
        "generated image size")
    flags.DEFINE_boolean('deep', False, 
        "use `deep` version of biggan or not")
    
    # choice of category and transformation
    flags.DEFINE_integer('y', 1, 
        "the category of the generated images")
    flags.DEFINE_enum('transform', 'horizontal_position', 
        ['horizontal_position', 'vertical_position', 'scale', 'brightness'], 
        "choice of transformation")
    
    app.run(get_barycenter)