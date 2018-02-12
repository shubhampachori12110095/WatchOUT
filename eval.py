from helper import *
flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "Number of batch size")
flags.DEFINE_integer("top_k", 5, "top_k")
#os.path.join(os.getcwd(), 'watchout/models/tripletnetwork/checkpoint')
flags.DEFINE_string('triplet_weights_dir', './watchout/models/tripletnetwork/checkpoint',
                           """Directory where triplet checkpoints live.""")
flags.DEFINE_string('transfer_values_dir', './watchout/data/transfer_values',
                           """Directory where transfer values live.""")
flags.DEFINE_string('image_dir', './watchout/data/raw_deepfashion_dataset/Img',
                           """Directory where transfer values live.""")
FLAGS = flags.FLAGS


def eval():
	queries = random_query_sample(FLAGS.batch_size)

	triplet_graph = tf.Graph()
	detection_graph = get_detector_graph()

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as d_sess:
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			d_tensors = {'image_tensor': image_tensor, 'detection_boxes': detection_boxes,
						 'detection_scores': detection_scores,
						 'detection_classes': detection_classes, 'num_detections': num_detections}
			with tf.Session(graph=triplet_graph) as triplet_sess:
				K.set_learning_phase(0)  # 1 : train, 0 : test
				base_model = get_triplet_model(is_full=False)
				full_model = get_triplet_model(is_full=True)
				triplet_sess.run(tf.global_variables_initializer())

				latest_weights, _ = get_latest_weights_and_global_step(FLAGS.triplet_weights_dir)
				full_model.load_weights(os.path.join(FLAGS.triplet_weights_dir,latest_weights))
				# model.compile(optimizer=Adam(lr=1e-7) ,loss=triplet_loss)

				base_model.set_weights(full_model.get_layer('base_inceptionv3').get_weights())

				for idx, query in enumerate(queries):
					raw_img, c_img, lbls = eval_data(query, d_sess, d_tensors)
					raw_img = np.expand_dims(raw_img, axis=0)
					print(str(idx)+' query image : '+query)
					print('retrievals with cropped image:')
					print(retrieve_images(triplet_sess, base_model, c_img, lbls[0][0]))
					print('retrievals with entire image:')
					print(retrieve_images(triplet_sess, base_model, raw_img, lbls[0][0]))


def main(argv=None):
	eval()


if __name__ == "__main__":
	tf.app.run()