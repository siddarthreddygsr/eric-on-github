import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.callbacks import ModelCheckpoint
import concurrent.futures

notes = []
def parse(mid):
	global notes
	midi = converter.parse(mid)

	print("Parsing %s" % mid)

	notes_to_parse = None

	try:
		s2 = instrument.partitionByInstrument(midi)
		notes_to_parse = s2.parts[0].recurse()
	except:
		notes_to_parse = midi.flat.notes

	for element in notes_to_parse:
		if isinstance(element, note.Note):
			notes.append(str(element.pitch))
		elif isinstance(element, chord.Chord):
			notes.append('.'.join(str(n) for n in element.normalOrder))

def main():
	
	global notes

	with concurrent.futures.ThreadPoolExecutor() as executor:
		results = [executor.map(parse, glob.glob("training-data/*.mid"))]

	with open('new-pickle/notes', 'wb') as filepath:
		pickle.dump(notes, filepath)

	n_vocab = len(set(notes))

	sequence_length = 100

	pitchnames = sorted(set(item for item in notes))

	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

	network_input = network_input / float(n_vocab)

	network_output = tensorflow.keras.utils.to_categorical(network_output)

	mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3"])
	with mirrored_strategy.scope():
		model = Sequential()
		model.add(LSTM(
			512,
			input_shape=(network_input.shape[1], network_input.shape[2]),
			recurrent_dropout=0.3,
			return_sequences=True
		))
		model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
		model.add(LSTM(512))
		model.add(BatchNorm())
		model.add(Dropout(0.3))
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(BatchNorm())
		model.add(Dropout(0.3))
		model.add(Dense(n_vocab))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	filepath = "new-models/epoch-{epoch:02d}-loss-{loss:.4f}-model.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == "__main__":
	main()
