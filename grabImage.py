import urllib2
import timeit
import time
import datetime

minutes = 15
seconds = 10 #minutes*60
numCycles = 4
hours = [7,13,17,19,21]
    
def grab(imageName):
	response = urllib2.urlopen('http://10.129.59.177/axis-cgi/jpg/image.cgi')
	with open(imageName, 'wb') as outfile:
		outfile.write(response.read())


def timeTest(N):
	secs = timeit.timeit("grabImage.grab('TimeitTest.jpg')",setup="import grabImage",number=N)
	print secs
	print 1.0*secs/N
	

	


def main():
	while True:
		t = time.time()
		timeStamp = datetime.datetime.fromtimestamp(t).strftime('%m%d%Y-%H%M%S') #time stamp
		h = time.strftime('%H')#the current hour
		fileName = 'WestLot ' + timeStamp +'.png' 
		#for i in xrange(numCycles):
		if h in hours:
			grab(fileName)
			print fileName
			time.sleep(seconds)
		
if __name__ == "__main__":
	main()
