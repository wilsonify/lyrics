import logging
from logging.config import dictConfig

from music21 import note, converter, serial, corpus, stream, analysis, environment

environment.set('musicxmlPath', '/usr/bin/musescore')
environment.set('midiPath', '/usr/bin/timidity')
environment.set('graphicsPath', '/usr/bin/gpicview')


def note_example():
    n = note.Note("D#3")
    n.duration.type = 'half'
    n.show()


def melody():
    little_melody = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
    little_melody.show()


def melody_midi():
    little_melody = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
    little_melody.show('midi')


def row_mat():
    print(serial.rowToMatrix([2, 1, 9, 10, 5, 3, 4, 0, 8, 7, 6, 11]))


def dicant_plot():
    dicant = corpus.parse('trecento/Fava_Dicant_nunc_iudei')
    dicant.plot('histogram', 'pitch')
    dicant.show()


def bwv():
    bwv295 = corpus.parse('bach/bwv295')
    for thisNote in bwv295.recurse().notes:
        thisNote.addLyric(thisNote.pitch.german)
    bwv295.show()


def catalog():
    opus_catalog = stream.Opus()
    for work in corpus.chorales.Iterator(1, 26):
        first_time_signature = work.parts[0].measure(1).getTimeSignatures()[0]
        if first_time_signature.ratioString == '3/4':
            incipit = work.measures(0, 2)
            opus_catalog.insert(0, incipit.implode())

    opus_catalog.show()


def patel():
    s = corpus.parse('AlhambraReel')
    analysis.patel.nPVI(s.flat)


def main():
    note_example()
    melody()
    melody_midi()
    row_mat()
    dicant_plot()
    bwv()
    catalog()
    patel()


if __name__ == "__main__":
    dictConfig({
        'version': 1,
        'formatters': {"simple": {"format": """%(asctime)s | %(name)s | %(lineno)s | %(levelname)s | %(message)s"""}},
        'handlers': {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
        'root': {"handlers": ["console"], "level": logging.DEBUG},
    })
    main()
