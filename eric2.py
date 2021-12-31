#v2.2
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
import multiprocessing
import concurrent.futures

notes = []
processes = []

def parse(file):

	global notes
	midi = converter.parse(file)

	print("Parsing %s" % file)

	notes_to_parse = None

	try: # file has instrument parts
		s2 = instrument.partitionByInstrument(midi)
		notes_to_parse = s2.parts[0].recurse()
	except: # file has notes in a flat structure
		notes_to_parse = midi.flat.notes

	for element in notes_to_parse:
		if isinstance(element, note.Note):
			notes.append(str(element.pitch))
		elif isinstance(element, chord.Chord):
			notes.append('.'.join(str(n) for n in element.normalOrder))



def main():

	global notes
	# for file in glob.glob("archive/Jazz-mid/*.mid"):
	# 	process = multiprocessing.Process(target=parse, args=[file])
	# 	process.start()
	# 	processes.append(process)
	print(multiprocessing.cpu_count())
	with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
		results = [executor.map(parse, glob.glob("archive/Jazz-mid/*.mid"))]

	# for process in processes:
	# 	process.join()

	print("notes:",notes)

	with open('notes', 'wb') as filepath:
		pickle.dump(notes, filepath)
	# get amount of pitch names
	n_vocab = len(set(notes))
	""" Prepare the sequences used by the Neural Network """
	sequence_length = 100

	# get all pitch names
	pitchnames = sorted(set(item for item in notes))

	 # create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / float(n_vocab)

	network_output = tensorflow.keras.utils.to_categorical(network_output)

	""" create the structure of the neural network """

	#creating an instance of the GPU
	#specifically uses the 8 GPUs of the NVIDIA DGX-1
	mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3","/gpu:4","/gpu:5","/gpu:6","/gpu:7"])

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

	# train the network
	filepath = "epoch-{epoch:02d}-loss-{loss:.4f}-model.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
	main()
