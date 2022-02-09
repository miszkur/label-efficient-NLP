import experiments.config as conf
import data_processing.ev_parser as ev
import tensorflow_text as text  # Registers the ops.
import model.classifier as mc 


def main():
  config = conf.multilabel_base()

  train_ds, _ = ev.create_dataset(batch_size=config.batch_size)
  val_ds, _ = ev.create_dataset(batch_size=config.batch_size, is_training=False, split='valid')
  test_ds, topics = ev.create_dataset(batch_size=config.batch_size, is_training=False, split='test')

  cls = mc.MultiLabelClassifier(config.bert)
  cls.compile(train_ds, epochs=5)
  cls.train(train_ds, val_ds, '.')

  cls.evaluate(test_ds, topics)

if __name__ == '__main__':
  main()


