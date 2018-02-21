import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd
import random,scipy, json
from PIL import Image
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input, Lambda
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

sys.path.append(os.path.join(os.getcwd(),"models/research"))
from object_detection.utils import label_map_util

MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(os.getcwd(), 'watchout/models/fasterRCNN/graph/frozen_inference_graph.pb')
PATH_TO_IMG = os.path.join(os.getcwd(),'watchout/data/raw_deepfashion_dataset/Img')


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


def get_latest_weights_and_global_step(_path):
	files = os.listdir(_path)
	files.reverse()
	return files[0], int(files[0].split('-')[1].split('.')[0])


def promising_box_index(scores):
	for idx, score in enumerate(scores):
		if score < 0.8:
			return idx
	return len(scores)


def promising_boxes(boxes, scores, is_eval=False):
	if is_eval:
		return tf.slice(tf.squeeze(boxes), [0, 0], [1, 4])
	promising_idx = promising_box_index(scores)
	return tf.slice(tf.squeeze(boxes), [0, 0], [promising_idx, 4])


def images_from_paths(targets, positives, negatives):
	assert len(targets) == len(positives) and len(positives) == len(negatives)
	t_list = []
	p_list = []
	n_list = []
	for t, p, n in zip(targets, positives, negatives):
		t_single, p_single, n_single = image_from_path(t, p, n)
		t_list.append(t_single)
		p_list.append(p_single)
		n_list.append(n_single)

	return np.array(t_list), np.array(p_list), np.array(n_list)


def image_from_path(target, positive, negative):
	tmp_a = Image.open(target)
	tmp_p = Image.open(positive)
	tmp_n = Image.open(negative)

	anchor = tmp_a.copy()
	anchor = anchor.resize((299, 299), Image.ANTIALIAS)
	anchor = np.array(anchor, dtype="float32")
	pos = tmp_p.copy()
	pos = pos.resize((299, 299), Image.ANTIALIAS)
	pos = np.array(pos, dtype="float32")
	neg = tmp_n.copy()
	neg = neg.resize((299, 299), Image.ANTIALIAS)
	neg = np.array(neg, dtype="float32")

	tmp_a.close()
	tmp_p.close()
	tmp_n.close()

	return anchor/255.0, pos/255.0, neg/255.0


def euclidean_distance(vects):
	anchor, target = vects
	# return K.sqrt(K.maximum(K.sum(K.square(anchor - target), axis=1, keepdims=True), K.epsilon()))
	return K.sum(K.square(anchor - target), axis=1, keepdims=True)


def triplet_loss(y_true, y_pred):
	margin = K.constant(0.2)
	return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))


def get_triplet_model(is_full=True):
	inceptionv3_input = Input(shape=(299, 299, 3))
	inceptionv3_f = InceptionV3(include_top=False, weights='imagenet', input_tensor=inceptionv3_input)
	net = inceptionv3_f.output
	net = Flatten(name='flatten')(net)
	net = Dense(1024, activation='relu', name='embedded')(net)

	base_model = Model(inceptionv3_f.input, net, name='base_inceptionv3')
	if is_full == False:
		return base_model
	input_anchor = Input(shape=(299, 299, 3), name='input_anchor')
	input_positive = Input(shape=(299, 299, 3), name='input_pos')
	input_negative = Input(shape=(299, 299, 3), name='input_neg')

	net_anchor = base_model(input_anchor)
	net_positive = base_model(input_positive)
	net_negative = base_model(input_negative)

	d_positive = Lambda(euclidean_distance, name='d_pos')(
		[net_anchor, net_positive])  # euclidean_distance((net_anchor, net_positive))
	d_negative = Lambda(euclidean_distance, name='d_neg')(
		[net_anchor, net_negative])  # euclidean_distance((net_anchor, net_negative))
	d_stacked = Lambda(lambda vects: K.stack(vects, axis=1), name='d_stacked')([d_positive, d_negative])

	triplet_model = Model([input_anchor, input_positive, input_negative], d_stacked, name='triplet_model')
	return triplet_model


def get_detector_graph():
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	return detection_graph


def eval_data(single_path, d_sess, d_tensors):
	image = Image.open(os.path.join('./watchout/data/raw_deepfashion_dataset/Img',single_path))
	image_np = load_image_into_numpy_array(image)
	image_np_expanded = np.expand_dims(image_np, axis=0)
	(_image_tensor, _boxes, scores, classes, num) = d_sess.run(
		[d_tensors['image_tensor'], d_tensors['detection_boxes'], d_tensors['detection_scores'],
		 d_tensors['detection_classes'], d_tensors['num_detections']],
		feed_dict={d_tensors['image_tensor']: image_np_expanded})
	target_box = promising_boxes(_boxes, scores[0], is_eval=True)
	cropped_image = tf.image.crop_and_resize(image=image_np_expanded / 255.0,
													 boxes=[target_box.eval().tolist()[0]],
													 box_ind=[0],
													 crop_size=[299, 299])
	return image.resize((299,299)), np.expand_dims(cropped_image[0].eval(), axis=0), classes


def get_train_data(_batch, d_sess,d_tensors):
	(t_lbl, t_path, p_path), (n_lbl, n_path) = random_triplet_sample(_batch)

	assert len(set([ x == y for (x,y) in zip(t_lbl, n_lbl)])) == 1 and list(set([ x == y for (x,y) in zip(t_lbl, n_lbl)]))[0] == False

	anchor, positive, negative = images_from_paths(t_path, p_path, n_path)

	for idx, single_batch in enumerate(t_path):

		image = Image.open(single_batch)
		image_np = load_image_into_numpy_array(image)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(_image_tensor, _boxes, scores, classes, num) = d_sess.run(
			[d_tensors['image_tensor'], d_tensors['detection_boxes'], d_tensors['detection_scores'],
			 d_tensors['detection_classes'], d_tensors['num_detections']],
			feed_dict={d_tensors['image_tensor']: image_np_expanded})

		target_boxes = promising_boxes(_boxes, scores[0])
		len_target_boxes = int(target_boxes.shape[0])

		cropped_images = []
		for box in target_boxes.eval().tolist():
			current_cropped_image = tf.image.crop_and_resize(image=image_np_expanded / 255.0,
															 boxes=[box],
															 box_ind=[0],
															 crop_size=[299, 299])
			cropped_images.append(current_cropped_image[0])
			anchor = np.append(anchor, np.expand_dims(current_cropped_image[0].eval(), axis=0), axis=0)
			positive = np.append(positive, np.expand_dims(positive[idx], axis=0), axis=0)
			negative = np.append(negative, np.expand_dims(negative[idx], axis=0), axis=0)

	assert anchor.shape[0] == positive.shape[0] and positive.shape[0] == negative.shape[0]

	return anchor, positive, negative


def random_triplet_sample(batch=1, is_train=True):
	ann = Anno(is_train=is_train)
	t_labels = []
	t_paths = []
	p_paths = []
	n_labels = []
	n_paths = []
	for i in range(batch):

		ran = random.randrange(0, len(ann))
		target_path, target_label = (ann.loc[ran]['image_name'], ann.loc[ran]['category_label'])
		positive_ran = random.choice(ann[ann['category_label'] == target_label].index.values)
		positive_path = ann[ann['category_label'] == target_label].loc[positive_ran]['image_name']

		while target_path == positive_path:
			print('loop with "target_path == positive_path"')
			positive_ran = random.choice(ann[ann['category_label'] == target_label].index.values)
			positive_path = ann[ann['category_label'] == target_label].loc[positive_ran]['image_name']

		negative_ran = random.choice(ann[ann['category_label'] != target_label].index.values)
		negative_path = ann[ann['category_label'] != target_label].loc[negative_ran]['image_name']
		negative_label = ann[ann['category_label'] != target_label].loc[negative_ran]['category_label']

		t_labels.append(target_label)
		t_paths.append(os.path.join(PATH_TO_IMG,target_path))
		p_paths.append(os.path.join(PATH_TO_IMG,positive_path))
		n_labels.append(negative_label)
		n_paths.append(os.path.join(PATH_TO_IMG,negative_path))

	return (t_labels, t_paths, p_paths), (n_labels, n_paths)


def Eval(head=True):
	path = os.path.join(os.getcwd(), 'watchout/data/raw_deepfashion_dataset/Eval/list_eval_partition.txt')
	data = pd.read_csv(path, sep=r"\s*", skiprows=[0], header=0)
	if head:
		return data.head(100)
	return data


def Anno(is_train=True, head=False):
	# Anno(os.path.join(os.getcwd(), 'watchout/data/raw_deepfashion_dataset/Anno'))
	category_path = os.path.join(os.getcwd(), 'watchout/data/raw_deepfashion_dataset/Anno/list_category_img.txt')
	# create_path(path,"list_category_img.txt")
	category_data = pd.read_csv(category_path, sep=r"\s*", skiprows=[0], header=0)
	eval_data = Eval(head=head)

	if is_train:
		category_data = pd.merge(category_data, eval_data[eval_data['evaluation_status'] == 'train'], how='inner',
								 on=['image_name'])
	else:
		category_data = pd.merge(category_data, eval_data[eval_data['evaluation_status'] == 'test'], how='inner',
								 on=['image_name'])
	category_data.drop('evaluation_status', 1)

	if head:
		return category_data.head(100)
	return category_data


def random_query_sample(batch=1):
	ann = Anno(is_train=False)
	paths = []
	for i in range(batch):
		ran = random.randrange(0, len(ann))
		paths.append(ann.loc[ran]['image_name'])
	return paths

def retrieve_images(t_sess, t_model, query_image, query_label, top_k=3):
	query_transfer_values = t_sess.run(t_model.layers[len(t_model.layers)-1].output,
									   feed_dict={'input_1:0':query_image})
	cos = []
	r_data = None
	with open('./watchout/data/transfer_values/label-'+str(int(query_label))+'.json', 'r') as infile:
		r_data = json.loads(infile.read())
	for key in r_data.keys():
		candidate = np.array(r_data[key])
		cos.append((key, np_cosine_distance(query_transfer_values,candidate)))
	cos.sort(key=lambda tup: tup[1])
	return cos[:top_k]


def np_cosine_distance(a,b):
    return (scipy.spatial.distance.cosine(a, b))