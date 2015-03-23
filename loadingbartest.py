import time
import sys
print('hello')
for i in range(100):
	sys.stdout.write("\r%d%%" %i)
	sys.stdout.flush()