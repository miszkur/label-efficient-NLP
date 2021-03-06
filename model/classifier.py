import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import model.bert_mapping as bm
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import classification_report
from official.nlp import optimization  # to create AdamW optimizer

tfk = tf.keras

class MultiLabelClassifier():
  def __init__(self, config):
    self.config = config
    self.model = self.build_model()

  def build_model(self):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    tfhub_handle_preprocess = bm.map_model_to_preprocess[self.config.bert_version]
    tfhub_handle_encoder = bm.map_name_to_handle[self.config.bert_version]

    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']
    # net = tf.keras.layers.Dropout(0.1)(net)
    # tf.estimator.MultiLabelHead(self.config.classes_num)
    net = tfk.layers.Dense(
      self.config.classes_num,
       activation=None, 
       name='classifier_head')(net)

    return tfk.Model(inputs=text_input, outputs=net)

  def compile(self, train_ds, epochs=5):
    self.epochs = epochs
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    optimizer = optimization.create_optimizer(
      init_lr=self.config.lr,
      num_train_steps=num_train_steps,
      num_warmup_steps=self.config.warmup_steps_num,
      optimizer_type='adamw')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [
      'binary_accuracy', # hamming_loss = 1 - accuracy
      tfa.metrics.F1Score(num_classes=self.config.classes_num, threshold=0.5)
    ]
    self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

  def train(self, train_ds, val_ds, save_path, show_history=True):
    checkpoint_filepath = os.path.join(save_path, 'checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    history = self.model.fit(
      x=train_ds,
      validation_data=val_ds,
      epochs=self.epochs,
      callbacks=[model_checkpoint_callback])

    model_save_path = os.path.join(save_path, 'model')
    self.model.save(model_save_path, include_optimizer=False)

    if show_history:
      history_dict = history.history
      self.plot_history(history_dict)

  def plot_history(self, history_dict):
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

  def evaluate(self, test_ds, target_names=None):
    print('Evaluation')
    self.model.evaluate(test_ds)

    print('Per-class performance')
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = self.model.predict(test_ds)
    y_pred = (y_pred > 0.5) 
    print(classification_report(y_true, y_pred, target_names=target_names))

    print('Per-class accuracy')
    correctly_classified = (y_pred == y_true)
    acc = correctly_classified.sum(axis=0) / y_pred.shape[0]
    for i, name in enumerate(target_names):
      print(f'{name}: {(acc[i]):.2f}')

  def load_model(self, saved_model_path):
      model = tf.keras.models.load_model(
          saved_model_path, compile=False)
      self.model = model
      return model