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
from tensorflow_consumer.translation.nmt import (
    download_data,
    preprocess_sentence,
    load_dataset,
    Encoder,
    Decoder,
)
from tensorflow_consumer.translation.phoneme2grapheme.phoneme2grapheme_training import (
    NUM_EXAMPLES,
    BATCH_SIZE,
    EMBEDDING_DIM,
    UNITS,
)


def main(sentence):
    path_to_file = os.path.join(DATA_DIR, "beatles_lyrics_combined", "grapheme2phoneme.txt")

    spa_tensor, eng_tensor, spa_lang, eng_lang = load_dataset(
        path_to_file, NUM_EXAMPLES
    )
    max_length_eng = eng_tensor.shape[1]
    max_length_spa = spa_tensor.shape[1]

    vocab_eng_size = len(eng_lang.word_index) + 1
    vocab_spa_size = len(spa_lang.word_index) + 1

    optimizer = tf.keras.optimizers.Adam()
    encoder = Encoder(vocab_eng_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_spa_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    checkpoint_dir = f"{CHECKPOINTS_DIR}/training_checkpoints/eng2spa"

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, encoder=encoder, decoder=decoder
    )

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    sentence = preprocess_sentence(sentence)

    inputs = [eng_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_eng, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)
    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([spa_lang.word_index["<start>"]], 0)

    for _ in range(max_length_spa):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )
        predicted_id = tf.argmax(predictions[0]).numpy()
        next_word = spa_lang.index_word[predicted_id]
        result += next_word + " "
        if next_word == "<end>":
            break
        dec_input = tf.expand_dims([predicted_id], 0)

    result_clean = result.replace("<start>", "").replace("<end>", "")
    return result_clean


if __name__ == "__main__":
    print(f"PATH = {os.getenv('PATH')}")
    print(f"LD_LIBRARY_PATH = {os.getenv('LD_LIBRARY_PATH')}")

    sentence1 = "IH1 T S V EH1 R IY0 K OW1 L D"
    result1 = main(sentence1)
    print(sentence1)
    print(result1)

    sentence2 = "B IY1 P BEEPM B IY1 P B IY1 P Y AE1"
    result2 = main(sentence2)
    print(sentence2)
    print(result2)

    sentence3 = "AA1 R DH EY1 S T IH1 L HH OW1 M"
    result3 = main(sentence3)
    print(sentence3)
    print(result3)

    sentence4 = "W AH1 T AY1 G AA1 T AH0 D UW1 T UW1 G EH1 T IH1 T TH R UW1 T UW1 Y UW1 IH1 M S UW2 P ER0 HH Y UW1 M AH0 N IH1 N AH0 V EY2 T IH0 V AH0 N D IH1 M M EY1 D AH1 V R AH1 B ER0"
    result4 = main(sentence4)
    print(sentence4)
    print(result4)
