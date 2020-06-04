import numpy as np
import tensorflow as tf
import keras
import tensorflow.keras.layers as L
import random
from random import choice
from keras.applications.inception_v3 import preprocess_input
import pickle
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import cv2

PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

class imageCaptioning:
  """
  The exact architecture of the decoder is subject to change because 
  I want to try different architectures on the train set. 
  However the essence of the model will remain the same as 
  explained in the Solution section. 
  It will be an ImageCaptioning class containing all the architecture:

  1. It will be able to train the model using the train_model method. 

  2. Save and restore the model once trained 

  3. Generate a caption for given image 

  4. Evaluate the model on the test set

  """

  def __init__(self, 
               vocab = None,
               lstm_size = 400, 
               word_embedding_size = 200, 
               prelogit_size = 600,
               prelstm_size = 120,
               restore_path = None
               ):
    """
    Initialization of the model's needed variables
    ---------------------------
    Parameters

    vocab - training vocabulary

    lstm_size - size of the lstm hidden states

    word_embedding_size - word embedding size

    prelogit_size - size of the layer that takes hidden states as input

    prelstm_sze - size of the layer that takes image emebdding as input,
    output then goes to initialize the lstm model

    restore_path - if given then the model restores all the needed variables
    from the the file given this path
    """

    # Encoding model that trasforms image to a embeddings
    self.encoding_model = keras.applications.InceptionV3(include_top = False)
    self.encoding_model=keras.models.Model(self.encoding_model.inputs,
                                          keras.layers.GlobalAveragePooling2D()(
                                               self.encoding_model.output))
    self.embedding_size = 2048

    if vocab is None:
      self.restore_model()
    elif restore_path is not None:
      self.restore_model(restore_path)
    else:

      # vocabulary
      self.vocab = vocab

      # lstm hidden state size
      self.lstm_size = lstm_size

      # word embedding size
      self.word_embedding_size = word_embedding_size

      # size of the layer in between lstm and the final logits
      self.prelogit_size = prelogit_size
      
      # size of the layer in between lstm initialization and image embedding
      self.prelstm_size = prelstm_size

      self.best_loss = None

      # list of layers
      self.layer_list = []

      # Parameter definition
      self.set_trainable_layers()
    

  def set_trainable_layers(self):
    """
    Defines all the layers in the model and stores them in a list
    """

    # Words -----> Word Emebeding
    self.word_to_embedding = L.Embedding(len(self.vocab['ind']),
                                  self.word_embedding_size)

    # Image embedding -----> One dense layer 
    self.img_to_prelstm = L.Dense(self.prelstm_size, 
                              input_shape=(None, self.embedding_size), 
                              activation='elu')
    
    # One Dense layer -----> Initial LSTM hidden states
    self.prelstm_to_lstm = L.Dense(self.lstm_size, 
                              input_shape=(None, self.prelstm_size), 
                              activation='elu')
    
    # Initial hidden states, Word Embeddings -----> LSTM hidde states
    self.lstm = L.LSTM(self.lstm_size, 
                       return_sequences=True, 
                       return_state = True)

    # LSTM hidden states -----> One dense layer(another one)
    self.lstm_to_prelogits = L.Dense(self.prelogit_size, 
                          input_shape=(None, self.lstm_size),
                          activation="elu")

    # One dense layer(another one) -----> Logits of the size of the vocabulary
    self.prelogits_to_logits = L.Dense(len(self.vocab['ind']),
                                  input_shape=(None, self.prelogit_size))
    
    # Adding all of those in order to save checkpoints 
    self.layer_list.append(self.word_to_embedding)
    self.layer_list.append(self.img_to_prelstm)
    self.layer_list.append(self.prelstm_to_lstm)
    self.layer_list.append(self.lstm)
    self.layer_list.append(self.lstm_to_prelogits)
    self.layer_list.append(self.prelogits_to_logits)
 
  def get_trainable_params(self):
    """
    Getting the list of all weights for tf2 gradient calculations
    """
    params = []
    for layer in self.layer_list:
      params += layer.weights
    return params

  
  def generate_batch(self, total_embeddings, total_captions, size):
    """
    Getting a random batch from the whole dataset
    """

    # Getting rnadon indices for the batch
    batch = random.sample(range(total_embeddings.shape[0]), size)

    # variables for image emebdding and captions in te current bathc
    self.batch_embeddings = np.array([total_embeddings[i] for i in batch])
    self.batch_captions = np.array([total_captions[i][random.choice(range(5))] 
                           for i in batch ])
    
  # function for caption generation process given one word get the next word
  def generate(self, h, c, sample = True):
    """
    Method which is used as forward pass during caption generation process
    at each time stamp 
    ----------------------------------------------------------------------
    Parameters

    h - previous lstm hidden vector to plug in

    c - previous lstm state vector to plug in

    sample - introduces two modes of generation. One (False) is where we 
    just take the most probable at each time stamp. Another one (True) 
    is when we taking a randon choice of next word based on the 
    probability distribution
    ----------------------------------------------------------------------
    Returns

    word_ind - predicted next word_index

    hidden_states[1] - current lstm hidden vector
    
    hidden_states[2] - current lstm state vector

    """

    # Word Emebeding
    word_embedding = self.word_to_embedding(self.batch_captions)

    # Next LSTM hidden state
    hidden_states = self.lstm(word_embedding, initial_state = [h, c])

    # Pre Logit Dense layer 
    prelogits = L.TimeDistributed(self.lstm_to_prelogits)(hidden_states[0])

    # Final Logits for the next word
    logits = self.prelogits_to_logits(prelogits)

    # Getting the Softmax probability distribution over the vocabulary
    probs = tf.nn.softmax(logits)
    probs = probs.numpy().ravel()
    probs = probs / np.sum(probs)

    # Getting the next word index
    word_ind = 0
    if sample:
      word_ind = np.random.choice(range(len(self.vocab['ind'])), p=probs)
    else:
      word_ind = np.argmax(probs)

    return word_ind, hidden_states[1], hidden_states[2]

  def forward_pass(self):
    """
    Performs the forward Pass and calculates the loss
    -------------------------------------------------
    Returns

    loss - the loss value
    """
  
    # Word embeddings
    word_embedding = self.word_to_embedding(self.batch_captions[:,:-1])
  

    # LSTM hidden states initialization
    lstm_init = self.prelstm_to_lstm(self.img_to_prelstm(self.batch_embeddings))

    # All the hidden states
    hidden_states = self.lstm(word_embedding, 
                              initial_state = [lstm_init, lstm_init])
    
    # PreLogits 
    prelogits = self.lstm_to_prelogits(hidden_states[0])
    
    # Final Logits
    logits = self.prelogits_to_logits(prelogits)
    
    # Ground Thruth to compare with our logits
    flat_ground_truth = tf.reshape(self.batch_captions[:,1:], (-1))
    
    # Making logits the same shape as ground truth 
    flat_logits = tf.reshape(logits, (-1, len(self.vocab['ind'])))
    
    # we want to skip training for PAD tokens
    flat_loss_mask = tf.math.not_equal(flat_ground_truth, 
                                       self.vocab['word'][PAD])
    
    # The loss function for each word
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth,
        logits=flat_logits
        )
    
    # Take average of all except PAD tokens
    return tf.reduce_mean(tf.boolean_mask(xent,flat_loss_mask))
    
  def get_grad(self, train = True):
    """
    Based on the loss get the weigth gradients
    ------------------------------------------
    Returns 

    grads - Gradients for all layers stored in self.layer_list

    loss - loss value

    trainable - list of trainable parameters
    """
    with tf.GradientTape() as tape:
      loss_value = self.forward_pass()
    if not train:
      return loss_value
    trainable = self.get_trainable_params()
    return tape.gradient(loss_value, trainable), loss_value, trainable

  def train_model(self, total_embeddings, total_captions, val_embeddings, 
                  val_captions, batch_size = 124, n_epochs = 16, 
                  batches_per_epoch = 1000, val_batch_per_epoch = 100):
    """
    Trains the model and prints out validation and test losses
    ----------------------------------------------------------
    Parameters

    total_embeddings - training numpy array for all the image emebeddings
    
    total_captions - training numpy array for all the image captions
    
    val_embeddings - validation numpy array for all the image embeddings
    
    val_captions - validation numpy array for all the image captions
    
    batch_size - the size of each batch
    
    n_epochs - number of epoch of training 
    
    batches_per_epoch - the amount of bathches to train per epoch 
    
    val_batch_per_epoch - the amount of bathches to validate per epoch
    """
     
    self.total_embeddings = total_embeddings
    self.total_captions =  total_captions
    self.val_embeddings = val_embeddings
    self.val_captions = val_captions

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # training loop
    for epoch in range(n_epochs):
      print("Epoch -- " + str(epoch))

      train_loss_total = 0
      val_loss_total = 0

      # Training =
      for batch_count in range(batches_per_epoch):

        # Generate batch
        self.generate_batch(total_embeddings, total_captions, batch_size)

        # Run this batch through the forward pass
        train_grads, train_loss, trainable = self.get_grad()

        # Using optimizer to perform backp pass
        optimizer.apply_gradients(zip(train_grads, trainable))

        # adding training loss to get average later
        train_loss_total += train_loss

        # print out progress
        if(batch_count % 50 == 0):
          print("*", end =" ") 

      # average training oss over all the batches per epoch
      train_loss_total /= batches_per_epoch
        
      # Validation
      for one_batch in range(val_batch_per_epoch):

        # Generate validation batch
        self.generate_batch(val_embeddings, val_captions, batch_size)

        # Get loss for validation
        val_loss_total += self.get_grad(train = False)
        
      # Average validation
      val_loss_total /= val_batch_per_epoch

      # print out the progress
      print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss_total, val_loss_total))

      print("Saving the epoch")
      self.save_model("epoch_" + str(epoch))

      # Keep the best validation loss value to save the best epoch
      if self.best_loss is None:
        self.best_loss = val_loss_total

      # if current epoch improves the validation loss then save the current 
      #epoch as the best 
      if val_loss_total <= self.best_loss:
        self.save_model( "best")

      print("------------------------------------------")

  # saving the model for chekpoints
  def save_model(self, name):
    """
    Saving the model per epoch
    --------------------------
    Parameters

    name - name of the save
    """

    #Openning the pickle file
    with open("checkpoints/" + name + '.pickle', 'wb') as f:

      # get the dictionary to save and save t
      pickle_dict = self.generate_pickle_dictionary()
      pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


  def generate_pickle_dictionary(self):
    """
    Put all the class variable to a dictionary to save it later
    -----------------------------------------------------------
    Returns

    pickle_dict - dictionary containing all the nessesary 
    variables to restore the class
    """
    
    pickle_dict = {}

    # all the variable we defiend in constructor
    pickle_dict['vocab'] = self.vocab
    pickle_dict['lstm_size'] = self.lstm_size
    pickle_dict['word_embedding_size'] = self.word_embedding_size
    pickle_dict['prelogit_size'] = self.prelogit_size
    pickle_dict['prelstm_size'] = self.prelstm_size
    pickle_dict['best_loss'] = self.best_loss
    pickle_dict['weights'] = []

    # Saving all the leyers' weights
    for layer in self.layer_list:
      weight_list = []
      for weight in layer.weights:
        weight_list.append(weight.numpy())
      pickle_dict['weights'].append(weight_list)

    return pickle_dict


  def restore_model(self, restore_path = "checkpoints/best.pickle"):
    """
    Restore the model given the backup path 
    ---------------------------------------
    Parameters

    restore_path - path to restore the whole
    thing. Default is to restore the model 
    with best validation loss
    """

    with open(restore_path,'rb') as f:

      pickle_dict = pickle.load(f)

      # Just repreating the constructor
      self.vocab = pickle_dict['vocab']
      self.lstm_size = pickle_dict['lstm_size']
      self.word_embedding_size = pickle_dict['word_embedding_size']
      self.prelogit_size = pickle_dict['prelogit_size']
      self.prelstm_size = pickle_dict['prelstm_size']
      self.best_loss = pickle_dict['best_loss']

      self.layer_list = []
      self.set_trainable_layers()

      # Setting all the layers in the model
      self.restore_layers(pickle_dict['weights'])

  def restore_layers(self, weight_list):
      """
      Restoring all the layers weights
      --------------------------------

      weight_list - list of weight saved
      in the pickle file 
      """

      # To set the weights hey need to be initialized first
      # I did not find the way to initialize except 
      # just running a dummy variables through them
      # and then setting the whole thing
      self.batch_embeddings = np.ones((1,2048))
      self.batch_captions = np.array([[0]])

      h = c = self.prelstm_to_lstm(self.img_to_prelstm(
        self.batch_embeddings))
      
      word_ind, h, c = self.generate(h, c)

      # So after the layer weights are initialized setting those weights
      for layer_num in range(len(self.layer_list)):
        self.layer_list[layer_num].set_weights(weight_list[layer_num])
  
  def get_caption(self, max_length, image = None, sample = False, 
                  embeddings = None):
    """
    Caption generation given the image
    ----------------------------------
    Parameters

    max_length - maximum length to
    generate

    image - source image if not given 
    then embeddings should

    sample - introduces two modes of 
    generation. One (False) is where we 
    just take the most probable at each 
    time stamp. Another one (True) 
    is when we taking a randon choice 
    of next word based on the 
    probability distribution

    embeddings - if image is not supported 
    then this should
    ----------------------------------
    Returns

    caption - the generated caption
    """

    # prepare image for encoder
    if embeddings is None:
      image = cv2.resize(image, (299,299))
      image = np.expand_dims(image, axis = 0)
      self.batch_embeddings = self.encoding_model.predict(
          preprocess_input(image))
    else:
      self.batch_embeddings = embeddings

    # set this batch 
    self.batch_captions = np.array([[self.vocab['word'][START]]])

    # initial LSTM hidden states
    current_h = current_c = self.prelstm_to_lstm(self.img_to_prelstm(
        self.batch_embeddings))

    caption = ""

    # Generate a word for ech time stamp
    for i in range(max_length):
      word_ind, h, c = self.generate(current_h, current_c, sample)
      
      if word_ind==self.vocab['word'][PAD] or word_ind==self.vocab['word'][END]:
        break

      # Adding the word to the caption 
      caption = caption + self.vocab['ind'][word_ind] 
      caption = caption + " "

      # Updating the LSTM hidden states
      current_h = h
      current_c = c

      # Updating the current caption 
      self.batch_captions = np.array([[word_ind]])
    return caption

  def get_BLEU(self, test_embeddings, test_captions_labels):
    """
    Go over each image in test set and get BLEU score over the whole set
    --------------------------------------------------------------------
    Parameters

    test_embeddings - test embeddings

    test_captions_labels - test caption list( list(each image) of 
    lists(each comment) of lists(each word but NOT tokenized))
    --------------------------------------------------------------------
    Returns

    score - test BLEU score
    """

    image_count = test_embeddings.shape[0]
    score = 0

    for image_ind in range(image_count):
      references = test_captions_labels[image_ind]
      max_length = max([len(one_sentence) for one_sentence in references])
      predictions = self.get_caption(max_length, 
                                     embeddings = test_embeddings[image_ind:
                                                                 image_ind+1])
      score += sentence_bleu(references, predictions.split())
    return score / image_count

