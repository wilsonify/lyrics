"""
trains a sequence to sequence (seq2seq) model for Phoneme to English (Grapheme) translation.

After training the model, you will be able to input a Phoneme sentence,
such as *""B EY1 B IY0 Y UW1 K AE1 N D R AY1 V M AY1 K AA1 R""*, and return the
English graphemes: *"Baby, you can drive my car"*

The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting.
This shows which parts of the input sentence has the model's attention while translating:

This example takes approximately 1 day to train on a single GeForce 660 GPU.
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_consumer.config import DATA_DIR, CHECKPOINTS_DIR
from tensorflow_consumer.translation.nmt import (
    preprocess_sentence,
    create_dataset,
    load_dataset,
    convert,
    Encoder,
    Decoder,
    BahdanauAttention,

)

NUM_EXAMPLES = 50000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
NUM_ATTENTION_UNITS = 10
EPOCHS = 10

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

        dec_input = tf.expand_dims([graph_lang.word_index["<start>"]] * BATCH_SIZE, 1)

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
    attention_plot = np.zeros((max_length_graph, max_length_phone))

    sentence = preprocess_sentence(sentence)

    inputs = [phone_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_phone, padding="post")
    inputs = tf.convert_to_tensor(inputs)
    result = ""

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([graph_lang.word_index["<start>"]], 0)

    for t in range(max_length_graph):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1,))  # storing the attention weights to plot later on
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        next_word = graph_lang.index_word[predicted_id]
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
    path_to_file = os.path.join(DATA_DIR, "beatles_lyrics_combined", "grapheme2phoneme.txt")
    if not os.path.isfile(path_to_file):
        print(f"cannot find data {path_to_file}. exit")
        sys.exit(1)

    grapheme_sentence = "Baby, you can drive my car"
    phoneme_sentence = "B EY1 B IY0 Y UW1 K AE1 N D R AY1 V M AY1 K AA1 R"
    print(preprocess_sentence(grapheme_sentence))
    print(preprocess_sentence(phoneme_sentence).encode("utf-8"))

    graph, phone = create_dataset(path_to_file, NUM_EXAMPLES)
    print(graph[-1])
    print(phone[-1])

    phone_tensor, graph_tensor, phone_lang, graph_lang = load_dataset(
        path_to_file, NUM_EXAMPLES
    )
    max_length_graph, max_length_phone = graph_tensor.shape[1], phone_tensor.shape[1]
    (
        phone_tensor_train,
        phone_tensor_val,
        graph_tensor_train,
        graph_tensor_val,
    ) = train_test_split(phone_tensor, graph_tensor, test_size=0.2)

    print(
        len(phone_tensor_train),
        len(graph_tensor_train),
        len(phone_tensor_val),
        len(graph_tensor_val),
    )

    print("spanish Language; index to word mapping")
    convert(phone_lang, phone_tensor_train[0])
    print()
    print("english Language; index to word mapping")
    convert(graph_lang, graph_tensor_train[0])

    BUFFER_SIZE = len(phone_tensor_train)

    steps_per_epoch = len(phone_tensor_train) // BATCH_SIZE

    vocab_phone_size = len(phone_lang.word_index) + 1
    vocab_graph_size = len(graph_lang.word_index) + 1
    dataset = tf.data.Dataset.from_tensor_slices(
        (phone_tensor_train, graph_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_phone_batch, example_graph_batch = next(iter(dataset))
    print(example_phone_batch.shape, example_graph_batch.shape)

    encoder = Encoder(vocab_phone_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_graph_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_phone_batch, sample_hidden)
    print(f"Encoder output shape: (batch size, sequence length, units) {sample_output.shape}")
    print(f"Encoder Hidden state shape: (batch size, units) {sample_hidden.shape}")

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print(f"Attention weights shape: (batch_size, sequence_length, 1) {attention_weights.shape}")

    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)

    print(f"Decoder output shape: (batch_size, vocab size) {sample_decoder_output.shape}")

    checkpoint_dir = f"{CHECKPOINTS_DIR}/training_checkpoints/spa2eng"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (spa, eng)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(spa, eng, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}")

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}")
        print(f"Time taken for 1 epoch {time.time() - start} sec\n")

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    translate("IH1 T S V EH1 R IY0 K OW1 L D")

    translate("B IY1 P BEEPM B IY1 P B IY1 P Y AE1")

    translate("AA1 R DH EY1 S T IH1 L HH OW1 M")

    translate(
        "W AH1 T AY1 G AA1 T AH0 D UW1 T UW1 G EH1 T IH1 T TH R UW1 T UW1 Y UW1 IH1 M S UW2 P ER0 HH Y UW1 M AH0 N IH1 N AH0 V EY2 T IH0 V AH0 N D IH1 M M EY1 D AH1 V R AH1 B ER0"
    )
