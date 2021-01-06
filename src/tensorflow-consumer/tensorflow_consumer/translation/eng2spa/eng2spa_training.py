"""
trains a sequence to sequence (seq2seq) model for Spanish to English translation.
This is an advanced example that assumes some knowledge of sequence to sequence models.

After training the model in this notebook, you will be able to input a Spanish sentence,
such as *"¿todavia estan en casa?"*, and return the English translation: *"are you still at home?"*

The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting.
This shows which parts of the input sentence has the model's attention while translating:

<img src="https://tensorflow.org/images/spanish-english.png" alt="spanish-english attention plot">

This example takes approximately 10 minutes to run on a single P100 GPU.
"""
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_consumer.config import DATA_DIR, CHECKPOINTS_DIR
from tensorflow_consumer.translation.nmt import (
    download_data,
    preprocess_sentence,
    create_dataset,
    load_dataset,
    convert,
    Encoder,
    Decoder,
    BahdanauAttention,
    BATCH_SIZE,
    EMBEDDING_DIM,
    UNITS,
    NUM_EXAMPLES,
    EPOCHS

)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none"
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
@tf.autograph.experimental.do_not_convert
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([spa_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)  # passing enc_output to the decoder
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)  # Teacher forcing - feeding the target as the next input

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(sentence):
    attention_plot = np.zeros((max_length_spa, max_length_eng))

    sentence = preprocess_sentence(sentence)

    inputs = [eng_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_eng, padding="post")
    inputs = tf.convert_to_tensor(inputs)
    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([spa_lang.word_index["<start>"]], 0)

    for t in range(max_length_spa):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1,))  # storing the attention weights to plot later on
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        next_word = spa_lang.index_word[predicted_id]
        result += next_word + " "
        if next_word == "<end>":
            break
        dec_input = tf.expand_dims([predicted_id], 0)  # the predicted ID is fed back into the model

    result_clean = result.replace("<start>", "").replace("<end>", "")
    return result_clean, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print(f"Input: {sentence}")
    print(f"Predicted translation: {result}")

    attention_plot = attention_plot[: len(result.split(" ")), : len(sentence.split(" "))]
    plot_attention(attention_plot, sentence.split(" "), result.split(" "))


if __name__ == "__main__":
    path_to_file = os.path.join(DATA_DIR, "spa-eng", "spa.txt")
    if not os.path.isfile(path_to_file):
        download_data()

    en_sentence = "May I borrow this book?"
    sp_sentence = "¿Puedo tomar prestado este libro?"
    print(preprocess_sentence(en_sentence))
    print(preprocess_sentence(sp_sentence).encode("utf-8"))

    en, sp = create_dataset(path_to_file, NUM_EXAMPLES)
    print(en[-1])
    print(sp[-1])

    spa_tensor, eng_tensor, spa_lang, eng_lang = load_dataset(path_to_file, NUM_EXAMPLES)
    max_length_eng = eng_tensor.shape[1]
    max_length_spa = spa_tensor.shape[1]
    (
        spa_tensor_train,
        spa_tensor_val,
        eng_tensor_train,
        eng_tensor_val,
    ) = train_test_split(
        spa_tensor,
        eng_tensor,
        test_size=0.2,
        random_state=None,
        shuffle=True,
        stratify=None,

    )

    print(
        len(spa_tensor_train),
        len(eng_tensor_train),
        len(spa_tensor_val),
        len(eng_tensor_val),
    )

    print("Spanish Language; index to word mapping")
    convert(spa_lang, spa_tensor_train[0])
    print()
    print("English Language; index to word mapping")
    convert(eng_lang, eng_tensor_train[0])

    BUFFER_SIZE = len(spa_tensor_train)

    steps_per_epoch = len(spa_tensor_train) // BATCH_SIZE

    vocab_spa_size = len(spa_lang.word_index) + 1
    vocab_eng_size = len(eng_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices(
        (spa_tensor_train, eng_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_spa_batch, example_eng_batch = next(iter(dataset))
    print(example_spa_batch.shape, example_eng_batch.shape)

    encoder = Encoder(vocab_eng_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_spa_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_eng_batch, sample_hidden)
    print(f"Encoder output shape: (batch size, sequence length, units) {sample_output.shape}")
    print(f"Encoder Hidden state shape: (batch size, units) {sample_hidden.shape}")

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print(f"Attention result shape: (batch size, units) {attention_result.shape}")
    print(f"Attention weights shape: (batch_size, sequence_length, 1) {attention_weights.shape}")

    sample_decoder_output, _, _ = decoder(tf.random.uniform(shape=(BATCH_SIZE, 1)), sample_hidden, sample_output)

    print(f"Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}")

    checkpoint_dir = f"{CHECKPOINTS_DIR}/training_checkpoints/eng2spa"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (spanish, english)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(
                inp=english,
                targ=spanish,
                enc_hidden=enc_hidden
            )
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}")

        if (epoch + 1) % 2 == 0:
            print("saving checkpoint every 2 epochs")
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}")
        print(f"Time taken for 1 epoch {time.time() - start} sec\n")

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    translate("it's very cold.")

    translate("this is my life.")

    translate("are they still home?")

    translate("try to find out.")
