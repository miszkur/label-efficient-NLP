import tensorflow as tf
import numpy as np
import bs4 as bs
import os

DATA_DIR = 'datasets'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024

def parse_xml_file(filename='Restaurants_Train_v2.xml'):
  data_path = os.path.join(DATA_DIR, filename)
  with open(data_path , 'r') as f:
    file = f.read() 
  file = bs.BeautifulSoup(file, "html.parser")

  sentences = file.find_all('sentence')
  topics = ['ambience', 'anecdotes/miscellaneous', 'food', 'price', 'service']

  reviews = []
  labels = []

  for review in sentences:
    reviews.append(review.find_all('text')[0].contents[0])

    aspect_categories = review.find_all('aspectcategory')
    categories_list = []
    for c in aspect_categories:
      categories_list.append(c['category'])

    current_label = []
    for topic in topics:
      if topic in categories_list:
        current_label.append(1)
      else:
        current_label.append(0)
    labels.append(current_label)

  return reviews, labels


def create_dataset(is_training=True):
  """Load and parse dataset.
  Args:
      filenames: list of image paths
      labels: numpy array of shape (BATCH_SIZE, N_LABELS)
      is_training: boolean to indicate training mode
  """
  
  reviews, labels = parse_xml_file()
  # Create a first dataset of file paths and labels
  dataset = tf.data.Dataset.from_tensor_slices((reviews, labels))
  # Parse and preprocess observations in parallel
  # dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
  
  if is_training == True:
      # This is a small dataset, only load it once, and keep it in memory.
      dataset = dataset.cache()
      # Shuffle the data each buffer size
      dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
      
  # Batch the data for multiple steps
  dataset = dataset.batch(BATCH_SIZE)
  # Fetch batches in the background while the model is training.
  dataset = dataset.prefetch(buffer_size=AUTOTUNE)
  
  return dataset
