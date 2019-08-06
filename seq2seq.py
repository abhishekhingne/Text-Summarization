import tensorflow as tf
from model import Model
from build_data import *
import time
from config import *


class TextSummarization:

    def __init__(self, num_hidden=150, num_layers=2, beam_width=10, embedding_size=300, learning_rate=1e-3,
                 batch_size=64, num_epochs=100, keep_prob=0.8, forward_only=False, with_model=False):
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.beam_width = beam_width
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.keep_prob = keep_prob
        self.forward_only = forward_only
        self.with_model = with_model
        self.data = BuildData()
        self.start_time = time.time()

    def start_training(self, train_x, train_y, word_dict, reversed_word_dict):
        tf.reset_default_graph()

        with tf.Session() as sess:
            model = Model(reversed_word_dict, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN, self.embedding_size, self.num_hidden,
                          self.num_layers, self.learning_rate, self.beam_width, self.keep_prob)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            batches = self.data.batch_iter(train_x, train_y, self.batch_size, self.num_epochs)
            num_batches_per_epoch = (len(train_x) - 1) // self.batch_size + 1

            print("\nTraining Started.")
            print("Number of batches per epoch :", num_batches_per_epoch)
            for batch_x, batch_y in batches:
                batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
                batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
                batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
                batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

                batch_decoder_input = list(
                    map(lambda d: d + (MAX_SUMMARY_LEN - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
                batch_decoder_output = list(
                    map(lambda d: d + (MAX_SUMMARY_LEN - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

                train_feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: batch_x,
                    model.X_len: batch_x_len,
                    model.decoder_input: batch_decoder_input,
                    model.decoder_len: batch_decoder_len,
                    model.decoder_target: batch_decoder_output
                }

                _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

                if step % 1000 == 0:
                    print("step {0}: loss = {1}".format(step, loss))

                if step % num_batches_per_epoch == 0:
                    hours, rem = divmod(time.time() - self.start_time, 3600)
                    minutes, seconds = divmod(rem, 60)
                    saver.save(sess, MODEL_PATH + "model.ckpt", global_step=step)
                    print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                          "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")

    def get_summary(self, x, reversed_word_dict):
        summary_text = []
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = Model(reversed_word_dict, MAX_ARTICLE_LEN, MAX_SUMMARY_LEN, self.embedding_size, self.num_hidden,
                          self.num_layers, self.learning_rate, self.beam_width, self.keep_prob,
                          forward_only=self.forward_only)
            try:
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
                saver.restore(sess, ckpt.model_checkpoint_path)
            except KeyError:
                print("Model path does'nt exists")
            batches = self.data.batch_iter(x, [0]*len(x), 1, 1)
            for batch_x, _ in batches:
                batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]
                feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: batch_x,
                    model.X_len: batch_x_len
                }
                prediction = sess.run(model.prediction, feed_dict=feed_dict)
                text = [[reversed_word_dict[y] for y in x] for x in prediction[:, 0, :]]
                summary = []
                for word in text[0]:
                    if word == '</s>':
                        break
                    if word not in summary:
                        summary.append(word)
                txt = " ".join(summary)
                summary_text.append(txt)
        return summary_text
