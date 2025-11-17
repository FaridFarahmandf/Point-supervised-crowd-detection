import os
import cv2
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import scipy
import scipy.spatial
#used to import .mat file
import scipy.io

############ locally-uniform distribution assumption (LUDA)##########
# For UCF-QNRF dataset ##########################
### set input dataset path for training or test
split = 'train' # train or test
root_dir = './data/UCF'

res_dataset = 'UCF'

if split == 'train':
    #directory of images for training
    img_path = os.path.join(root_dir, 'Train')
    #directory of annotation info - .mat files are in the same directory
elif split == 'test':
    img_path = os.path.join(root_dir, 'Test')
    #directory of annotation info - .mat files are in the same directory

if not os.path.exists(os.path.join('data/cache', res_dataset)):
    os.makedirs(os.path.join('data/cache', res_dataset))

res_path = os.path.join('data/cache', res_dataset, split)

# UCF dataset parameters (similar to ShanghaiTech part_A)
range_split = 10.
ratio = 1.
max_sp = None

img_names = [x for x in os.listdir(img_path) if x.endswith('.jpg')]    #read all jpg filenames
image_data = []               #will be used at the end--to write files
weights_list = []
scales = []
valid_count = 0          #number of valid images--with at least 1 person
img_count = 0           #number of all images
box_count = 0            #number of all heads in one image

for img_name in img_names:      #read all images
    img_count += 1              #record the total number of imgs
    if img_count % 20 == 0:
        print(img_count)
    full_img_path = os.path.join(img_path, img_name)   #full image path
    img = cv2.imread(full_img_path)                   #read imgs
    if img is None:
        print('Warning: Could not read image:', full_img_path)
        continue
    img_height, img_width = img.shape[:2]           #img-height=row of the img;img_width=colums of the img
    
    # UCF mat files have format: img_XXXX_ann.mat
    mat_name = img_name.replace('.jpg', '_ann.mat')
    mat_path = os.path.join(img_path, mat_name)
    
    if not os.path.exists(mat_path):
        print('Warning: Annotation file not found:', mat_path)
        continue
        
    data = scipy.io.loadmat(mat_path)  #find path of .mat file
    # UCF uses 'annPoints' instead of 'image_info'
    if 'annPoints' in data:
        centers = data['annPoints'].astype(np.float32)   #read all the centers in the image
    else:
        print('Warning: annPoints not found in', mat_path)
        continue

    box_count_image = centers.shape[0]     #number of boxes(targets) in each img
    box_count += box_count_image           #number of boxes in the whole dataset

    if box_count_image > 0:                #judge wether it is a valid img:if no less than 1 box->valid
        valid_count += 1
        annotation = {}                  #to restore annotation info
        annotation['filepath'] = full_img_path             #to restore full imgpath into the annotation[]
        # add scale in bboxes height
        #KD tree
        tree = scipy.spatial.KDTree(centers.copy(), leafsize=1024) #intialization of the tree
        k = min(box_count_image, 3)              #for each box, find the nearest k-1 boxes around it(it itself is included in k->so k-1)
        if k <= 2:                                          #need parameters-img_height,centers,len(boxes)
            scale = max(img_height / (4. + k), 12)
            scale = np.ones(box_count_image) * scale
            scale_weight = 0.1 if k==1 else 1.
            scale_weight = np.ones(box_count_image) * scale_weight
        else:
            crowd_range = np.max(centers[:,1]) - np.min(centers[:,1]) # range: y_max - y_min;for the whole img,find the gap of y between the highest box and the lowest box
            circle_scale = crowd_range / range_split                          #initialization of scale of local window
            distances, distances_idx = tree.query(centers, k=min(box_count_image//2, box_count_image))  #distances:Euclinear distance between box X and others(by order from near to far)
            #distances_idx is the coresponding index of these distances
            distances_mean = ratio * np.mean(distances[:,1:k],axis=1)      #mean of distances
            places = np.where(distances <= circle_scale)     #how many boxes within circle_scale
            unique, counts = np.unique(places[0], return_counts=True) # places[0]: row index
            #counts:how many boxes within circle_scale
            take_d_places = dict(zip(unique, counts))    #zip:make (unique,counts)
            # Initialize scale and scale_weight for all points
            scale = np.zeros(box_count_image)
            scale_weight = np.zeros(box_count_image)
            for key,value in take_d_places.items():
                idx_in_circle = distances_idx[key, :value]
                s_p = np.mean(distances_mean[idx_in_circle])
                if max_sp is not None:
                    s_p = np.clip(s_p, 2, max_sp)         #set the number of s_P to be 2-max_sp
                else:
                    s_p = np.clip(s_p, 2, None)         #set the number of s_P to be 2-infinite
                scale[key] = s_p                 #scale->s_P
                scale_weight[key] = value
            # For points not in take_d_places, use their own distances_mean
            for i in range(box_count_image):
                if scale[i] == 0:
                    s_p = distances_mean[i]
                    if max_sp is not None:
                        s_p = np.clip(s_p, 2, max_sp)
                    else:
                        s_p = np.clip(s_p, 2, None)
                    scale[i] = s_p
                    scale_weight[i] = 1.0

        weights_list.extend(list(scale_weight))
        scales.extend(list(scale))
        boxes_with_scale = np.zeros((box_count_image,4),dtype=np.float32)         #initialization of boxes_with_scale,datatype:int64
        boxes_with_scale[:, 0], boxes_with_scale[:, 2] = centers[:, 0] - scale / 2., centers[:, 0] + scale / 2. #x1, x2
        boxes_with_scale[:, 1], boxes_with_scale[:, 3] = centers[:,1] - scale/2., centers[:,1] + scale/2. #y1, y2
        boxes_with_scale[:, 0:4:2] = np.clip(boxes_with_scale[:, 0:4:2], 0, img_width - 1)
        boxes_with_scale[:, 1:4:2] = np.clip(boxes_with_scale[:, 1:4:2], 0, img_height - 1)
        annotation['bboxes'] = boxes_with_scale                                          #store results in annotation[]
        annotation['confs'] = 0.6 * np.ones((boxes_with_scale.shape[0]))
        annotation['w_bboxes'] = scale_weight
        annotation['ignoreareas'] = np.zeros((0, 4), dtype=np.float32)  # UCF doesn't have ignore areas
        image_data.append(annotation)

weights = np.array(weights_list,dtype='float')        #for testing:record weights_max,weights_mean
print('weights_max: {}'.format(weights.max()))
print('weights_mean: {}'.format(weights.mean()))
print('weights_std: {}'.format(weights.std()))

scales = np.array(scales,dtype='float')
print('scales_max: {}'.format(scales.max()))
print('scales_min: {}'.format(scales.min()))
print('scales_mean: {}'.format(scales.mean()))
print('scales_std: {}'.format(scales.std()))

for image_data_i in image_data:
    image_data_i['w_bboxes'] = np.clip(image_data_i['w_bboxes'], None, 50)

print('{} images and {} valid images and {} boxes'.format(img_count, valid_count,box_count))
with open(res_path, 'wb') as fid:                                                    #to write a file in the res_path
    pickle.dump(image_data, fid, pickle.HIGHEST_PROTOCOL)

