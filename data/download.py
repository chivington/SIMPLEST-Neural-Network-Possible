import requests as rq
import os, sys, time

def download_data():
	have_train = os.path.exists('mnist_train.csv')
	have_test = os.path.exists('mnist_test.csv')
	if have_train and have_test:
		print(f'\n MNIST data already downloaded.')
	else:
		print('\n Downloading MNIST data...')
		train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
		test_url = 'https://pjreddie.com/media/files/mnist_test.csv'
		for url in [train_url, test_url]:
			f = url.split('/')[-1]
			sys.stdout.write(f'  - {f}')
			start = time.time()
			req = rq.get(url)
			res = req.text
			fp = open(f, 'w')
			fp.write(res)
			fp.close
			end = time.time()
			sys.stdout.write(f' ({round(end - start, 2)}s)\n')

download_data()
