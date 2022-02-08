import ml_collections

DATA_DIR = 'datasets'
RESULTS_DIR = 'results'

def ev_dataset_config():
  config = ml_collections.ConfigDict()
  config.data_dir = DATA_DIR
  config.classes_num = 8
  return config

def bert_config():
  config = ml_collections.ConfigDict()
  config.bert_version = 'small_bert/bert_en_uncased_L-2_H-128_A-2'
  config.lr = 5e-5 # 1e-4
  config.weight_decay = 0.01
  config.warmup_steps_num = 500
  return config

def multilabel_base():
  config = ml_collections.ConfigDict()
  config.results_dir = RESULTS_DIR
  config.epochs = 5  # 20

  config.bert = bert_config()

  dataset_info = ev_dataset_config()
  config.classes_num = dataset_info.classes_num
  config.data_dir = dataset_info.data_dir
  return config
