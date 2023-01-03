import numpy as np
import matplotlib.pyplot as plt
import time, sys, os

# ----- SET MATH ENVIRONMENT
MATH_ENV = 'numpy'
blas = np
try:
	import cupy as cu
	MATH_ENV, blas = 'cupy', cu
except Exception as e:
	print(f" CuPy not found, running neural network on CPU.\n To install CuPy, visit:\n  https://docs.cupy.dev/en/stable/install.html")

blas.random.seed(4)

# ----- DATA FUNCTIONS
def one_hot(Y, classes):
	encoded = blas.zeros((Y.shape[0], classes))
	for i in range(Y.shape[0]):
		encoded[i][int(Y[i][0])] = 1
	return encoded

def shuffle(X, Y):
	idxs = blas.array([i for i in range(X.shape[0])])
	blas.random.shuffle(idxs)
	return X[idxs], Y[idxs]

def load_data():
	os.system('cls' if os.name == 'nt' else 'clear')
	if os.path.exists('data/mnist_train.csv') and os.path.exists('data/mnist_test.csv'):
		print(f'\n Loading training & testing datasets...')
		files = ['mnist_train', 'mnist_test']
		out = []
		for file in files:
			sys.stdout.write(f'  - {file}')
			load_start = time.time()
			data = np.loadtxt(f'data/{file}.csv', delimiter = ',')
			x = data[:,1:] / data[:,1:].max()
			y = one_hot(data[:,:1], 10)
			if MATH_ENV == 'cupy': x, y = cu.array(x), cu.array(y)
			load_end = time.time()
			out.append((x, y))
			print(f' ({round(load_end - load_start, 2)}s)')
		return out[0][0], out[0][1], out[1][0], out[1][1]
	else:
		url = 'https://pjreddie.com/media/files/'
		print(f' Datasets not downloaded. Download at:\n  - {url}mnist_train.csv\n  - {url}mnist_test.csv')
		sys.exit()

def batch_data(X, Y, batch_size, cycles):
	sys.stdout.write(f'\n Batching training dataset... ')
	batching_start = time.time()
	train_batches = []
	for e in range(cycles):
		shuffled_X, shuffled_Y = shuffle(X, Y)
		m = X.shape[0]
		num_batches = m // batch_size
		batches = []
		for batch in range(num_batches - 1):
			start = batch * batch_size
			end = (batch + 1) * batch_size
			x, y = X[start:end], Y[start:end]
			batches.append((x, y))
		last_start = num_batches * batch_size
		batches.append((X[last_start:], Y[last_start:]))
		train_batches.append(batches)
	batching_end = time.time()
	print(f'({blas.around(batching_end - batching_start, 2)}s)    ')
	return train_batches

# ----- METRICS FUNCTIONS
def plot_lines(test_acc, data):
	data = [{'title': t, 'data':d} for t,d in [('Cost', data[0]), ('Accuracy', data[1]), ('Time', data[2])]]
	fig, plots = plt.subplots(3)
	plt.suptitle(f'Model Performance Metrics (Model Acc. {test_acc}%)', fontsize=16, fontweight='bold')
	fig.subplots_adjust(top=0.89, bottom=0.13, left=0.12, right=0.96, hspace=0.25, wspace=0.01)
	for i, p in enumerate(plots):
		plot_data, plot_title = data[i]["data"], data[i]["title"]
		if MATH_ENV == 'cupy': plot_data = plot_data.get()
		lbls = [
			f'{blas.around(plot_data[0] - plot_data[-1], 4)} Cost Decrease',
			f'{test_acc}% Test Accuracy',
			f'Avg. {blas.around(blas.mean(plot_data), 2)}s Cycle Duration'
		]
		p.plot(range(1, len(plot_data) + 1), plot_data, label=lbls[i], linewidth=0.75)
		for l in ['x', 'y']: p.tick_params(axis=l, rotation=45)
		p.set(ylabel=plot_title)
		p.margins(x=0.03, y=0.05)
		p.legend(loc='best')
		if plot_title == 'Time': p.set(xlabel='Cycle')
		else: p.set_xticks([])
	plt.xticks(np.arange(1, len(data[0]['data'])+1, 1))
	plt.show()

def show_predictions(test_imgs, predictions, model_acc):
	idxs = np.random.randint(0, test_imgs.shape[0], size=15)
	imgs = test_imgs[idxs]
	preds = blas.argmax(predictions[idxs], axis=1)
	if MATH_ENV == 'cupy': imgs = imgs.get()
	imgs = imgs.reshape([15, 28, 28])
	fig, axs = plt.subplots(3, 5)
	plt.suptitle(f' MNIST Model Predictions (Model Acc. {model_acc}%)', fontsize=16, fontweight='bold')
	fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.75)
	for row in range(3):
		for col in range(5):
			p = axs[row,col]
			p.set_title(f'Prediction: {preds[row * 3 + col]}')
			p.imshow(imgs[row * 3 + col], interpolation='nearest')
			p.set_xticks([])
			p.set_yticks([])
	plt.show()

# ----- NEURAL NETWORK CLASSES
class Net:
	def __init__(self, X, Y, x, y, layers=[256,128], cycles=3, lr=0.001, batch_size=64):
		print(f'\n Initializing network... (cycles={cycles}, learning rate={lr})')
		self.train_x = X
		self.train_y = Y
		self.test_x = x
		self.test_y = y
		self.layers = self.init_layers(layers)
		self.cycles = cycles
		self.lr = lr
		self.print_freq = round(cycles * 0.1) if round(cycles * 0.1) > 0 else 1
		self.batch_size = batch_size

	def init_layers(self, layers):
		init = []
		num_layers = len(layers)
		n = self.train_x.shape[1]
		for l in range(num_layers + 1):
			if l < num_layers:
				layer = layers[l]
				input_size = n if l == 0 else layers[l-1]
				output_size = layer
				init.append(Dense([input_size, output_size]))
				init.append(Softplus())
			else:
				init.append(Dense([layers[-1], 10]))
				init.append(Softmax())
			print(f'  Layer {l+1} Dimensions: ({init[-2].weights.shape[0]} x {init[-2].weights.shape[1]})')
		return init

	def forward(self, batch):
		output = batch
		for layer in self.layers:
			output = layer.forward(output)
		return output

	def backward(self, batch, error):
		grad = (1/batch.shape[0]) * error
		for layer in list(reversed(self.layers)):
			grad = layer.backward(grad, self.lr)

	def train(self):
		m, n = self.train_x.shape
		costs, accs, times = blas.array([]), blas.array([]), blas.array([])
		batches = batch_data(self.train_x, self.train_y, self.batch_size, self.cycles)
		batch_size = batches[0][0][0].shape[0]
		print(f'\n TRAINING...')
		train_start = time.time()
		for cycle in range(self.cycles):
			if (cycle > 0) and (accs[-1] >= 100.0): break
			print_cycle = True if ((cycle==0) or ((cycle+1)%self.print_freq==0) or (cycle==self.cycles-1)) else False
			if print_cycle: sys.stdout.write(f' {f" {cycle+1}/{self.cycles} >> ":>12}')
			cycle_start = time.time()
			cost, acc = 0, 0
			for b,batch in enumerate(batches[cycle]):
				if cycle==0 and b==31: cycle_start = train_start = time.time()
				output = self.forward(batch[0])
				error = output - batch[1]
				self.backward(batch[0], error)
				cost += blas.mean(error**2)
				acc += blas.count_nonzero(blas.argmax(output, axis=1) == blas.argmax(batch[1], axis=1)) / batch[0].shape[0]
			cycle_end = time.time()
			costs, accs, times = blas.append(costs, (cost/len(batches[cycle]))), blas.append(accs, (acc*100/len(batches[cycle]))), blas.append(times, (cycle_end-cycle_start))
			if print_cycle: print(f'{f"Duration: {blas.around(times[-1], 2)}s":<15} / {f"Accuracy: {blas.around(accs[-1], 5)}%"}')
		train_end = time.time()
		train_time = blas.around(train_end - train_start, 2)
		train_mins = int((train_time) // 60)
		train_secs = int((train_time) - (train_mins * 60))
		avg_time = blas.around(blas.average(times), 2)
		times, accs, costs, acc_delta = blas.around(times, 2), blas.around(accs, 5), blas.around(costs, 5), blas.around(accs[-1] - accs[0], 2)
		print(f'\n TOTAL TRAINING TIME: {train_mins}m : {train_secs}s\n AVG. CYCLE TIME: {avg_time}s')
		return [costs, accs, times, train_time, avg_time, acc_delta]

	def test(self):
		print(f'\n TESTING...')
		x, y = self.test_x, self.test_y
		output = self.forward(x)
		acc = blas.around(100 * blas.count_nonzero(blas.argmax(output, axis=1) == blas.argmax(y, axis=1)) / x.shape[0], 5)
		print(f'   TEST ACCURACY: {acc}%')
		return acc, output

class Layer:
	def __init__(self):
		self.input = None
		self.output = None
		self.type = 'BASE LAYER CLASS'

	def forward(self, input):
		pass

	def backward(self, grad):
		pass

class Dense(Layer):
	def __init__(self, weights=[10,10]):
		self.type = 'Dense'
		self.weights = blas.random.randn(weights[0], weights[1]) * blas.sqrt(2.0/weights[0])

	def forward(self, input):
		self.input = input
		self.output = input.dot(self.weights)
		return self.output

	def backward(self, grad, lr):
		dW = blas.dot(grad.T, self.input).T
		dZ = blas.dot(grad, self.weights.T)
		self.weights -= dW * lr
		return dZ

class Softplus(Layer):
	def __init__(self):
		self.type = 'Softplus'

	def forward(self, z):
		self.input = z
		self.output = blas.log(1.0 + blas.exp(z))
		return self.output

	def backward(self, grad, lr):
		ez = blas.exp(self.input)
		return ez / (1.0 + ez) * grad

class Softmax(Layer):
	def __init__(self):
		self.type = 'Softmax'

	def forward(self, z):
		self.input = z
		z = z - blas.max(z, axis=1).reshape(z.shape[0], 1)
		ez = blas.exp(z)
		self.output = ez / blas.sum(ez, axis=1).reshape(z.shape[0], 1)
		return self.output

	def backward(self, grad, lr):
		return grad

if __name__ == "__main__":
	train_x, train_y, test_x, test_y = load_data()

	layers = [64,32]
	cycles = 5
	lr = 0.007

	nn = Net(train_x, train_y, test_x, test_y, layers, cycles, lr)
	stats = nn.train()
	test_acc, predictions = nn.test()

	plot_lines(test_acc, [stats[0], stats[1], stats[2]])
	show_predictions(test_x, predictions, test_acc)
