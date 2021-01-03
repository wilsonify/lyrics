"""
based on https://github.com/martin-gorner/tensorflow-rnn-shakespeare
"""

import glob
import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow_consumer import config
import tarfile

def encode(phoneme_str):
    """
    encode phoneme as a one hot encoded tensor
    :param phoneme_str:
    :return:
    """
    lines_list = phoneme_str.split("\n")
    integer_encoded = []
    for line in lines_list:
        integer_encoded.append(config.CHAR_TO_INT['\n'])
        phoneme_list = line.split(" ")
        for phoneme in phoneme_list:
            try:
                integer_encoded.append(config.CHAR_TO_INT[phoneme])
            except KeyError:
                continue
    logging.debug("%r", "integer_encoded = {}".format(integer_encoded))
    return tf.one_hot(integer_encoded, config.ALPHASIZE, on_value=1.0, off_value=0.0, axis=-1)


def decode(one_hot_tensor):
    """
    decode tensor to phoneme
    :param one_hot_tensor:
    :return:
    """
    indexes = tf.argmax(one_hot_tensor, axis=1)
    phoneme_str = ""
    for index in indexes:
        logging.info(int(index))
        try:
            phoneme_str += " " + config.INT_TO_CHAR[int(index)]
        except KeyError:
            continue
    logging.info(phoneme_str)
    return phoneme_str


def convert_from_alphabet(character):
    # logging.debug("convert_from_alphabet")
    """Encode a character
    # Specification of the supported alphabet (subset of ASCII-7)
    # 10 line feed LF
    # 32-64 numbers and punctuation
    # 65-90 upper-case letters
    # 91-97 more punctuation
    # 97-122 lower-case letters
    # 123-126 more punctuation

    :param character: one character
    :return: the encoded value
    """
    if character == 9:
        return 1
    if character == 10:
        return 127 - 30  # LF
    elif 32 <= character <= 126:
        return character - 30
    else:
        return 0  # unknown


def convert_to_alphabet(code_point, avoid_tab_and_lf=False):
    """Decode a code point
    # encoded values:
    # unknown = 0
    # tab = 1
    # space = 2
    # all chars from 32 to 126 = c-30
    # LF mapped to 127-30

    :param code_point: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    logging.debug("convert_to_alphabet")
    if code_point == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if code_point == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= code_point + 30 <= 126:
        return code_point + 30
    else:
        return 0  # unknown


def encode_text(string_to_encode):
    """Encode a string.
    :param string_to_encode: a text string
    :return: encoded list of code points
    """
    result = list(map(lambda a: convert_from_alphabet(ord(a)), string_to_encode))
    return result


def decode_to_text(encoded_str, avoid_tab_and_lf=False):
    """Decode an encoded string.
    :param encoded_str: encoded list of code points
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return:
    """
    logging.debug("decode_to_text")
    return "".join(map(lambda a: chr(convert_to_alphabet(a, avoid_tab_and_lf)), encoded_str))


def sample_from_probabilities(probabilities, topn=config.ALPHASIZE):
    """
    Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities.
    If topn is specified, only the topn highest probabilities are taken into account.
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    logging.debug("sample_from_probabilities")
    probabilities_squeeze = np.squeeze(probabilities)
    probabilities_squeeze[np.argsort(probabilities_squeeze)[:-topn]] = 0
    probabilities_squeeze = probabilities_squeeze / np.sum(probabilities_squeeze)
    return np.random.choice(config.ALPHASIZE, 1, p=probabilities_squeeze)[0]


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    logging.debug("rnn_minibatch_sequencer")
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(
        data[0:rounded_data_len], [batch_size, nb_batches * sequence_size]
    )
    ydata = np.reshape(
        data[1: rounded_data_len + 1], [batch_size, nb_batches * sequence_size]
    )

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size: (batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size: (batch + 1) * sequence_size]
            x = np.roll(
                x, -epoch, axis=0
            )  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch


def find_book(index, bookranges):
    logging.debug("find_book")
    return next(
        book["name"] for book in bookranges if (book["start"] <= index < book["end"])
    )


def find_book_index(index, bookranges):
    logging.debug("find_book_index")
    return next(
        i for i, book in enumerate(bookranges) if (book["start"] <= index < book["end"])
    )


def print_learning_learned_comparison(
        x_input,
        y_output,
        losses,
        bookranges,
        batch_loss,
        batch_accuracy,
        epoch_size,
        index,
        epoch,
):
    """
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510


    :param x_input:
    :param y_output:
    :param losses:
    :param bookranges:
    :param batch_loss:
    :param batch_accuracy:
    :param epoch_size:
    :param index:
    :param epoch: # epoch_size in number of batches
    :return:
    """
    logging.debug("print_learning_learned_comparison")
    logging.debug("""Display utility for printing learning statistics""")

    batch_size = x_input.shape[0]  # batch_size in number of sequences
    sequence_len = x_input.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    epoch_string = ""
    formatted_bookname = ""
    decx = ""
    decy = ""
    loss_string = ""
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decode_to_text(x_input[k], avoid_tab_and_lf=True)
        decy = decode_to_text(y_output[k], avoid_tab_and_lf=True)
        bookname = find_book(index_in_epoch, bookranges)
        formatted_bookname = "{: <10.40}".format(bookname)  # min 10 and max 40 chars
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
        logging.info(print_string.format(decx, decy, loss_string))
        index += sequence_len
    format_string = "└{:─^" + str(len(epoch_string)) + "}"
    format_string += "{:─^" + str(len(formatted_bookname)) + "}"
    format_string += "┴{:─^" + str(len(decx) + 2) + "}"
    format_string += "┴{:─^" + str(len(decy) + 2) + "}"
    format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
    footer = format_string.format(
        "INDEX", "BOOK NAME", "TRAINING SEQUENCE", "PREDICTED SEQUENCE", "LOSS"
    )

    logging.info("print statistics")
    logging.info(footer)
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
    stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(
        batch_string, batch_loss, batch_accuracy
    )
    logging.info("\n")
    logging.info("TRAINING STATS: %s", stats)


class Progress:
    """
    Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""

    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        logging.info("initialize Progress")
        self.maxi = maxi
        self.current_progress = self.__start_progress(
            maxi
        )()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.current_progress)

    def __print_header(self):
        logging.info("\n")
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        logging.info(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            diff_y_x = dy - dx
            for _ in range(maxi):
                k = 0
                while diff_y_x >= 0:
                    print("=", end="", flush=True)
                    k += 1
                    diff_y_x -= dx
                diff_y_x += dy
                yield k

        return print_progress


def read_data_files(glob_pattern, validation=True):
    """
    Read data files according to the specified glob pattern
    Optionnaly set aside the last file as validation data.
    No validation data is returned if there are 5 files or less.
    :param glob_pattern: for example "data/*.txt"
    :param validation: if True (default), sets the last file aside as validation data
    :return: training data, validation data, list of loaded file names with ranges
     If validation is
    """
    logging.info("read_data_files")
    codetext = []
    bookranges = []
    for text_file_path in glob.glob(glob_pattern):
        logging.info("Loading file %s", text_file_path)
        with open(text_file_path, "r") as text_file:
            text_contents = text_file.read()
            logging.debug("len(text_contents) = %d", len(text_contents))
            text_encoded = encode_text(text_contents)
            start = len(codetext)
            codetext.extend(text_encoded)
            end = len(codetext)
            bookranges.append(
                {"start": start, "end": end, "name": text_file_path.rsplit("/", 1)[-1]}
            )
    logging.debug("bookranges = {}".format(bookranges))
    if not bookranges:
        sys.exit("No training data has been found. Aborting.")

    logging.debug("""
    # For validation, use roughly 90K of text,
    # but no more than 10% of the entire text
    # and no more than 1 book in 5 => no validation at all for 5 files or fewer.
    """)
    logging.debug("10% of the text is how many files?")
    total_len = len(codetext)
    validation_len = 0
    nb_books1 = 0
    for book in reversed(bookranges):
        validation_len += book["end"] - book["start"]
        nb_books1 += 1
        if validation_len > total_len // 10:
            break

    logging.debug("# 90K of text is how many books ?")
    validation_len = 0
    nb_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"] - book["start"]
        nb_books2 += 1
        if validation_len > 90 * 1024:
            break

    logging.debug("20% of the books is how many books ?")
    nb_books3 = len(bookranges) // 5

    logging.debug("# pick the smallest")
    nb_books = min(nb_books1, nb_books2, nb_books3)

    if nb_books == 0 or not validation:
        cutoff = len(codetext)
    else:
        cutoff = bookranges[-nb_books]["start"]
    valitext = codetext[cutoff:]
    codetext = codetext[:cutoff]
    return codetext, valitext, bookranges


def print_data_stats(datalen, valilen, epoch_size):
    """
    print some stats about the data
    :param datalen:
    :param valilen:
    :param epoch_size:
    :return:
    """
    logging.debug("print_data_stats")
    datalen_mb = datalen / 1024.0 / 1024.0
    valilen_kb = valilen / 1024.0
    logging.info(
        "Training text size is {:.2f}MB with {:.2f}KB set aside for validation. There will be {} batches per epoch".format(
            datalen_mb,
            valilen_kb,
            epoch_size
        )
    )


def print_validation_header(validation_start, bookranges):
    """
    print some info about validation
    :param validation_start:
    :param bookranges:
    :return:
    """
    logging.debug("print_validation_header")
    bookindex = find_book_index(validation_start, bookranges)
    books = ""
    for i in range(bookindex, len(bookranges)):
        books += bookranges[i]["name"]
        if i < len(bookranges) - 1:
            books += ", "
    logging.info("Validating on %s", "{: <60}".format(books))


def print_validation_stats(loss, accuracy):
    """
    print some stats in a nice format
    :param loss:
    :param accuracy:
    :return:
    """
    logging.debug("print_validation_stats")
    logging.info(
        "VALIDATION STATS: loss: {:.5f}, accuracy: {:.5f}".format(
            loss,
            accuracy
        ))


def print_text_generation_header():
    """
    print some text at the beginning
    :return:
    """
    logging.debug("print_text_generation_header")
    logging.info("\n")
    logging.info("┌{:─^111}┐ Generating random text from learned state")


def print_text_generation_footer():
    """
    print some text at the end
    :return:
    """
    logging.debug("print_text_generation_footer")
    logging.info("\n")
    logging.info("└{:─^111}┘ End of generation")


def frequency_limiter(n_occurances, multiple=1, modulo=0):
    """

    :param n_occurances:
    :param multiple:
    :param modulo:
    :return:
    """
    logging.debug("frequency_limiter")

    def limit(i):
        return i % (multiple * n_occurances) == modulo * multiple

    return limit


def unzip(file_name, destination_dir):
    extractor = tarfile.open
    compression_type = 'gz'
    read_mode = "r:{}".format(compression_type)
    if file_name.endswith(".tar.gz"):
        extractor = tarfile.open
        compression_type = 'gz'
        read_mode = "r:{}".format(compression_type)
    elif file_name.endswith(".tar"):
        extractor = tarfile.open
        compression_type = ''
        read_mode = "r:{}".format(compression_type)
    elif file_name.endswith(".zip"):
        import zipfile
        extractor = zipfile.ZipFile
        read_mode = 'r'

    open_args = [file_name, read_mode]
    with extractor(*open_args) as ref:
        ref.extractall(path=destination_dir)