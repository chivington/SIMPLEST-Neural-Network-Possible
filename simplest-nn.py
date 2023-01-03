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
	for i in range(Y.shape[0]): encoded[i][int(Y[i][0])] = 1
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
		p.tick_params(axis='x', rotation=60)
		# for l in ['x', 'y']: p.tick_params(axis=l, rotation=45)
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

# ----- NEURAL NETWORK FUNCTIONS
def init_weights(layers, n):
	init = []
	for l,layer in enumerate(layers + [10]):
		input_size = n if l == 0 else layers[l-1]
		init.append(blas.random.randn(input_size, layer) * blas.sqrt(2.0/input_size))
		print(('\n' if l==0 else '') + f' Weights {l+1} Dimensions: {init[-1].shape}')
	return init

def forward(input, weights):
	outputs = []
	for i,w in enumerate(weights[:-1]):
		outputs.append([input.dot(w) if i==0 else outputs[-1][1].dot(w)])
		outputs[-1].append(blas.log(1.0 + blas.exp(outputs[-1][0])))
	outputs.append([outputs[-1][1].dot(weights[-1])])
	ez = blas.exp(outputs[-1][0] - blas.max(outputs[-1][0], axis=1).reshape(outputs[-1][0].shape[0], 1))
	outputs[-1].append(ez / blas.sum(ez, axis=1).reshape(ez.shape[0], 1))
	return outputs

def backward(batch, error, weights, forward_pass, lr):
	grad = (1/batch.shape[0]) * error
	for i,l in enumerate(reversed(forward_pass[:-1])):
		dW = blas.dot(grad.T, l[1]).T
		dZ = blas.dot(grad, weights[-(i+1)].T)
		weights[-(i+1)] -= dW * lr
		ez = blas.exp(l[0])
		dA = ez / (1.0 + ez) * dZ
		grad = dA
	weights[0] -= blas.dot(grad.T, batch).T * lr

def train(train_x, train_y, layers, cycles, lr):
	m, n = train_x.shape
	costs, accs, times = blas.array([]), blas.array([]), blas.array([])
	weights, batches = init_weights(layers, 784), batch_data(train_x, train_y, 64, cycles)
	print_freq = round(cycles*0.1) if round(cycles*0.1) > 0 else 1
	print(f'\n TRAINING...')
	train_start = time.time()
	for cycle in range(cycles):
		if (cycle>0) and (accs[-1]>=100.0): break
		print_cycle = True if ((cycle==0) or ((cycle+1)%print_freq==0) or (cycle==cycles-1)) else False
		if print_cycle: sys.stdout.write(f' {f" {cycle+1}/{cycles} >> ":>12}')
		cycle_batches = batches[cycle]
		cycle_start = time.time()
		cost, acc = 0, 0
		for b,batch in enumerate(cycle_batches):
			if cycle==0 and b==31: cycle_start = train_start = time.time()
			output = forward(batch[0], weights)
			error = output[-1][1] - batch[1]
			backward(batch[0], error, weights, output, lr)
			cost += blas.mean(error**2)
			acc += blas.count_nonzero(blas.argmax(output[-1][1], axis=1) == blas.argmax(batch[1], axis=1)) / batch[0].shape[0]
		cycle_end = time.time()
		costs, accs, times = blas.append(costs,(cost/len(cycle_batches))), blas.append(accs,(acc*100/len(cycle_batches))), blas.append(times,(cycle_end-cycle_start))
		if print_cycle: print(f'{f"Duration: {blas.around(times[-1], 2)}s":<15} / {f"Accuracy: {blas.around(accs[-1], 5)}%"}')
	train_end = time.time()
	train_time = blas.around(train_end - train_start, 2)
	train_mins = int((train_time) // 60)
	train_secs = int((train_time) - (train_mins * 60))
	avg_time = blas.around(blas.average(times), 2)
	times, accs, costs = blas.around(times,2), blas.around(accs,5), blas.around(costs,5)
	train_acc_delta = blas.around(accs[-1]-accs[0], 2)
	print(f'\n TOTAL TRAINING TIME: {train_mins}m : {train_secs}s\n AVG. CYCLE TIME: {avg_time}s')
	return [costs, accs, times, train_time, avg_time, train_acc_delta, weights]

def test(x, y, weights):
	predictions = forward(x, weights)[-1][1]
	acc = blas.around(100 * blas.count_nonzero(blas.argmax(predictions, axis=1) == blas.argmax(y, axis=1)) / x.shape[0], 5)
	print(f'\n TEST ACCURACY: {acc}%')
	return acc, predictions

if __name__ == "__main__":
	train_x, train_y, test_x, test_y = load_data()

	layers = [256,128,256,1024]
	cycles = 31
	learning_rate = 0.007

	stats = train(train_x, train_y, layers, cycles, learning_rate)
	test_accuracy, predictions = test(test_x, test_y, stats[-1])

	plot_lines(test_accuracy, [stats[0], stats[1], stats[2]])
	show_predictions(test_x, predictions, test_accuracy)
