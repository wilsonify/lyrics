"""
# + [markdown] id="CiwtNgENbx2g"
# This notebook trains a sequence to sequence (seq2seq) model for Spanish to English translation. This is an advanced example that assumes some knowledge of sequence to sequence models.
#
# After training the model in this notebook, you will be able to input a Spanish sentence, such as *"¿todavia estan en casa?"*, and return the English translation: *"are you still at home?"*
#
# The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting. This shows which parts of the input sentence has the model's attention while translating:
#
# <img src="https://tensorflow.org/images/spanish-english.png" alt="spanish-english attention plot">
#
# Note: This example takes approximately 10 minutes to run on a single P100 GPU.
"""
import os

import tensorflow as tf
from tensorflow_consumer.config import DATA_DIR, CHECKPOINTS_DIR
from tensorflow_consumer.nmt import (
    download_data,
    preprocess_sentence,
    load_dataset,
    Encoder,
    Decoder,
    BATCH_SIZE,
    EMBEDDING_DIM,
    UNITS,
    NUM_EXAMPLES,

)


def main(sentence):
    path_to_file = os.path.join(DATA_DIR, "spa-eng", "spa.txt")
    if not os.path.isfile(path_to_file):
        download_data()

    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, NUM_EXAMPLES)
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    optimizer = tf.keras.optimizers.Adam()
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    checkpoint_dir = f"{CHECKPOINTS_DIR}/training_checkpoints"

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding="post")
    inputs = tf.convert_to_tensor(inputs)
    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]], 0)

    for _ in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + " "
        dec_input = tf.expand_dims([predicted_id], 0)

    result_clean = result.replace("<start>", "").replace("<end>", "")
    return result_clean


if __name__ == "__main__":
    sentence1 = "hace mucho frio aqui."
    result1 = main(sentence1)
    print(sentence1)
    print(result1)

    sentence2 = "esta es mi vida."
    result2 = main(sentence2)
    print(sentence2)
    print(result2)

    sentence3 = "¿todavia estan en casa?"
    result3 = main(sentence3)
    print(sentence3)
    print(result3)

    sentence4 = "trata de averiguarlo."
    result4 = main(sentence4)
    print(sentence4)
    print(result4)
