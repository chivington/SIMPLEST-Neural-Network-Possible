import numpy as np
import cupy as cu
import matplotlib.pyplot as plt
import time, sys, os

# SEED RNG
cu.random.seed(4)

# DATA FUNCTIONS
def one_hot(Y, classes):
	encoded = cu.zeros((Y.shape[0], classes))
	for i in range(Y.shape[0]): encoded[i][int(Y[i,0])] = 1
	return encoded

def load_data():
	os.system('cls' if os.name == 'nt' else 'clear')
	if os.path.exists('../data/mnist_train.csv') and os.path.exists('../data/mnist_test.csv'):
		print(f'\n Loading training & testing datasets...')
		files = ['mnist_train', 'mnist_test']
		out = []
		for file in files:
			sys.stdout.write(f'  - {file}')
			load_start = time.time()
			data = np.loadtxt(f'../data/{file}.csv', delimiter = ',')
			x = cu.array((data[:,1:] - data[:,1:].mean()) / data[:,1:].max())
			y = cu.array(one_hot(data[:,:1], 10))
			load_end = time.time()
			out.append((x, y))
			print(f' ({round(load_end - load_start, 2)}s)')
		return out[0][0], out[0][1], out[1][0], out[1][1]
	else:
		print(f' Datasets not downloaded. To download, change into the data directory and run: "python download.py"')
		sys.exit()

def batch_data(X, Y, cycles, batch_size=64):
	m = X.shape[0]
	num_batches = m // batch_size
	batches = []
	sys.stdout.write(f'\n Batching training data...')
	batching_start = time.time()
	for cycle in range(cycles):
		cycle_batches = []
		for batch in range(num_batches):
			start = batch * batch_size
			end = (batch + 1) * batch_size
			if end > m: end = m - 1
			x, y = X[start:end], Y[start:end]
			cycle_batches.append((x, y))
		batches.append(cycle_batches)
	batching_end = time.time()
	sys.stdout.write(f'({round(batching_end - batching_start, 2)}s)\n')
	return batches

# METRICS FUNCTIONS
def plot_lines(times, costs, accuracies):
	fig, plots = plt.subplots(3)
	plt.suptitle(f'Model Performance Metrics', fontsize=13, fontweight='bold')
	fig.subplots_adjust(top=0.91, bottom=0.13, left=0.12, right=0.96, hspace=0.25, wspace=0.01)
	conf = [
		('Cost', f'{round(costs[0] - costs[-1], 5)} Cost Decrease', '#e77', costs),
		('Accuracy', f'{accuracies[-1]}% Train Accuracy', '#7e7', accuracies),
		('Time', f'Avg. {round(sum(times) / len(times), 2)}s Cycle Duration', '#77e', times)
	]
	for i, p in enumerate(plots):
		y, title, lbl = conf[i][3], conf[i][0], conf[i][1]
		p.plot(range(1, len(costs)+1), y, label=lbl, linewidth=0.75, color=conf[i][2])
		p.tick_params(axis='x', rotation=60)
		p.set(ylabel=title)
		p.margins(x=0.03, y=0.05)
		p.legend(loc='best')
		p.set_xticks([])
	plt.xticks(range(1, len(costs)+1, round(cycles//10/10)*10 or cycles//10 or 1))
	plt.show()

def show_predictions(tst_imgs, predictions, model_acc):
	idxs = np.random.randint(0, tst_imgs.shape[0], size=15)
	imgs = tst_imgs[idxs].get()
	preds = cu.argmax(predictions[idxs], axis=1)
	imgs = imgs.reshape([15, 28, 28])
	fig, axs = plt.subplots(3, 5)
	plt.suptitle(f' MNIST Model Predictions\n (Test Acc. {model_acc}%)', fontsize=16, fontweight='bold')
	fig.subplots_adjust(top=0.83, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.75)
	for row in range(3):
		for col in range(5):
			p = axs[row,col]
			p.set_title(f'Prediction: {preds[row * 3 + col]}')
			p.imshow(imgs[row * 3 + col], interpolation='nearest')
			p.set_xticks([])
			p.set_yticks([])
	plt.show()

# NEURAL NETWORK CLASSES
class Net:
	def __init__(self, X, Y, x, y, layers=[32,64], cycles=3, lr=0.005, bs=64):
		print(f'\n Initializing network...')
		self.trnx = X
		self.trny = Y
		self.tstx = x
		self.tsty = y
		self.layers = self.init_layers([X.shape[1]] + layers + [Y.shape[1]])
		self.cycles = cycles
		self.lr = lr
		self.print_freq = round(cycles//10/10)*10 or cycles//10 or 1
		self.bs = bs
		self.indent = 9 if cycles > 999 else (7 if cycles > 99 else 5)

	def init_layers(self, layers):
		init = []
		for l, layer in enumerate(layers[:-1]):
			init.append(Dense([layer, layers[l+1]]))
			init.append(Softplus() if l < len(layers)-2 else Softmax())
			print(f'  - Layer {l+1}: {init[-2].weights.shape}')
		return init

	def forward(self, X):
		predictions = X
		for layer in self.layers:
			predictions = layer.forward(predictions)
		return predictions

	def backward(self, error, lr):
		prev_grad = error
		for l, layer in enumerate(reversed(self.layers[:-1])):
			prev_grad = layer.backward(prev_grad, lr)

	def train(self):
		(m, n), k = self.trnx.shape, self.trny.shape[1]
		training_batches = batch_data(self.trnx, self.trny, self.cycles, self.bs)
		times, costs, accuracies = [], [], []
		print(f'\n Training model...\n  - cycles:{self.cycles}, learning rate:{self.lr}, batch size:{self.bs}\n')
		trn_start = time.time()
		for c in range(self.cycles):
			if (c > 0) and (costs[-1] <= 0): break
			prnt_cyc = (c==0) or ((c+1) % self.print_freq == 0) or (c==self.cycles-1)
			if prnt_cyc: sys.stdout.write(f'  >> {f"{c+1}/{self.cycles}":<{self.indent}}')
			cyc_start = time.time()
			cost, accuracy, batches = 0, 0, training_batches[c]
			for b, (x,y) in enumerate(batches):
				predictions = self.forward(x)
				accuracy += self.calc_accuracy(predictions, y)
				error = predictions - y
				cost += np.mean(error**2)
				self.backward(error, self.lr)
				if c==0 and b==35: trn_start = cyc_start = time.time()
			costs.append(round(float(cost) / len(batches), 6))
			accuracies.append(round(float(accuracy) / len(batches) * 100, 6))
			times.append(round(time.time() - cyc_start, 2))
			if prnt_cyc: print(f' Time: {f"{times[-1]}s":<5} | Cost: {f"{costs[-1]}":<8} | Acc: {f"{accuracies[-1]}%"}')
		trn_t = time.time() - trn_start
		print(f'\n TRAINING TIME: {int(trn_t//60)}m : {int(trn_t-int(trn_t//60)*60)}s\n AVG. CYCLE TIME: {round(trn_t/len(times),2)}s')
		return times, costs, accuracies

	def test(self):
		predictions = self.forward(self.tstx)
		accuracy = np.around(self.calc_accuracy(predictions, self.tsty) * 100, 6)
		print(f'\n TEST ACCURACY: {accuracy}%')
		return accuracy, predictions

	def calc_accuracy(self, A, Y):
		return np.count_nonzero(np.argmax(A, axis=1) == np.argmax(Y, axis=1)) / Y.shape[0]

class Dense:
	def __init__(self, weights=[10,10]):
		self.type = 'Dense'
		self.weights = cu.random.randn(weights[0], weights[1]) * cu.sqrt(2.0/weights[0])

	def forward(self, input):
		self.input = input
		self.output = input.dot(self.weights)
		return self.output

	def backward(self, grad, lr):
		dW = self.input.T.dot(grad)
		dX = grad.dot(self.weights.T)
		self.weights -= dW * lr
		return dX

class Softplus:
	def __init__(self):
		self.type = 'Softplus'

	def forward(self, z):
		self.input = z
		self.output = cu.log(1.0 + cu.exp(z))
		return self.output

	def backward(self, grad, lr):
		ez = cu.exp(self.input)
		return ez / (1.0 + ez) * grad

class Softmax:
	def __init__(self):
		self.type = 'Softmax'

	def forward(self, z):
		self.input = z
		ez = cu.exp(z)
		self.output = ez / cu.sum(ez, axis=1).reshape(z.shape[0], 1)
		return self.output


if __name__ == "__main__":
	trnx, trny, tstx, tsty = load_data()

	layers = [64,128]
	cycles = 31
	learning_rate = 0.007
	batch_size = 64

	nn = Net(trnx, trny, tstx, tsty, layers, cycles, learning_rate, batch_size)
	times, costs, accuracies = nn.train()
	tst_accuracy, predictions = nn.test()

	plot_lines(times, costs, accuracies)
	show_predictions(tstx, predictions, tst_accuracy)
