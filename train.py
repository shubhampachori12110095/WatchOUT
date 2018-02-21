import os
import sys
from keras.optimizers import Adam
sys.path.append(os.path.join(os.getcwd(),"models/research"))
from helper import *

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "Number of batch size")
flags.DEFINE_string('triplet_weights_dir', './watchout/models/tripletnetwork/checkpoint',
                           """Directory where triplet checkpoints live.""")
flags.DEFINE_string('image_dir', './watchout/data/raw_deepfashion_dataset/Img',
                           """Directory where transfer values live.""")
FLAGS = flags.FLAGS

def train():
    detection_graph = get_detector_graph()
    triplet_graph = tf.Graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            d_tensors = {'image_tensor': image_tensor, 'detection_boxes': detection_boxes,
                         'detection_scores': detection_scores,
                         'detection_classes': detection_classes, 'num_detections': num_detections}

            with tf.Session(graph=triplet_graph) as triplet_sess:
                global_step = tf.get_variable('global_step', initializer=0, trainable=False)
                increment_global_step = tf.assign(global_step, global_step + 1)

                triplet_model = get_triplet_model()
                triplet_sess.run(tf.global_variables_initializer())

                try:
                    latest_weights, global_step_value = get_latest_weights_and_global_step(FLAGS.triplet_weights_dir)
                    triplet_model.load_weights(os.path.join(FLAGS.triplet_weights_dir,latest_weights))
                    triplet_sess.run(global_step.assign(global_step_value))
                    print('latest weights loaded, global_step='+str(global_step_value))
                except Exception:
                    print('No weights file!')

                triplet_model.compile(optimizer=Adam(lr=1e-7), loss=triplet_loss)

                train_writer = tf.summary.FileWriter('./watchout/models/tripletnetwork/logs/train', triplet_sess.graph)
                test_writer = tf.summary.FileWriter('./watchout/models/tripletnetwork/logs/test')

                while True:
                    anchor, positive, negative = get_train_data(_batch=FLAGS.batch_size, d_sess=sess, d_tensors=d_tensors)

                    final_batch = anchor.shape[0]

                    training_loss = triplet_model.train_on_batch(x=[anchor, positive, negative],
                                                                 y=np.random.randint(2, size=(1, 2, final_batch)).T)
                    triplet_sess.run(increment_global_step)

                    if global_step.eval() % 200 == 0:
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss",
                                                                           simple_value=training_loss)])
                        train_writer.add_summary(train_summary, global_step.eval())
                        triplet_model.save_weights(os.path.join(os.getcwd(),
                                                                'watchout/models/tripletnetwork/checkpoint/weights-' + str(
                                                                    global_step.eval()) + '.hdf5'))
                        print('periodic train write done')

                        if global_step.eval() % 600 == 0:
                            _anchor, _positive, _negative = get_train_data(_batch=FLAGS.batch_size, d_sess=sess,
                                                                           d_tensors={'image_tensor': image_tensor,
                                                                                      'detection_boxes': detection_boxes,
                                                                                      'detection_scores': detection_scores,
                                                                                      'detection_classes': detection_classes,
                                                                                      'num_detections': num_detections})
                            final_test_batch = _anchor.shape[0]
                            test_loss = triplet_model.test_on_batch(x=[_anchor, _positive, _negative],
                                                                    y=np.random.randint(2, size=(1, 2, final_test_batch)).T)
                            test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss",
                                                                              simple_value=test_loss)])
                            test_writer.add_summary(test_summary, global_step.eval())
                            print('periodic test write done')
def main(argv=None):
    train()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    tf.app.run()