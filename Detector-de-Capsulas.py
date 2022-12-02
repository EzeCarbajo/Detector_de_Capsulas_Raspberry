import cv2
import numpy as np

#Asignación de variables globales
path = "/home/pi/Desktop/Tesis/Resources/VideoYFotogramas/Vid_10_1.jpg"#"/home/pi/Desktop/Tesis/Resources/Prueba7.jpg" #"/home/pi/Desktop/Tesis/Resources/Prueba1.jpg" #

umbralGris = 20
umbralMin = 157
umbraMax = 255
countArchivo = 0
anchoLinea = 3
anchoView = 15
altoView = 15

def loadValores():
	img = cv2.imread(path)
	imgTest = cv2.imread(path)
	ancho = img.cols
	alto = img.rows
	plantillaBebedero = np.full((alto,ancho), [255,255,255])

def dentroBebedero(plantilla, x, y):
	return (plantilla[y,x] > 0)


def getPorcentajeCapsulas(imgParm, umbral):
	totalPixeles = imgParm.cols*imgParm.rows
	pixelesCapsulas = 0
	for row in img:
		for pixel in row:
			if (pixel >= umbral):
				pixelesCapsulas += 1

	return pixelesCapsulas / totalPixeles


def getContornoMayor(contornos):
	contornoMayor = 0
	areaMayor = contourArea(contornos[0])
	for contorno in contornos:
		nuevaArea = contourArea(contorno)
		if (nuevaArea > areaMayor):
			areaMayor = nuevaArea
			contornoMayor = contorno
	
	return contornoMayor

def GetCuadrantesCubiertos(mitadX, mitadY, line):
	cuadrantes = [false, false, false, false]
	if (line[0] < mitadX):
		if (line[1] < mitadY):
			cuadrantes[1] = true
		else:
			cuadrantes[2] = true
	else:
		if (line[1] < mitadY):
			cuadrantes[0] = true
		else:
			cuadrantes[3] = true
	if (line[2] < mitadX):
		if (line[3] < mitadY):
			cuadrantes[1] = true
		else:
			cuadrantes[2] = true
	else:
		if (line[3] < mitadY):
			cuadrantes[0] = true
		else:
			cuadrantes[3] = true
	count = 0
	for cuadrante in cuadrantes:
		if cuadrante:
			count+=1
	
	return count, cuadrantes

def CheckIfExisteEn(arr, valoresARevisar):
	for i, arrVal in enumerate(arr):
		if (arrVal == valoresARevisar[i]):
			return true
	
	return false

def GetIfDosLineasCubrenCadaCuadrante(lines, mitadX, mitadY):
	cuadrantes1 = [ false, false, false, false ]
	cuadrantes2 = [ false, false, false, false ]
	
	i = 0
	while(i < lines.size()):
		cantCuadrantes1, cuadrantes1 = GetCuadrantesCubiertos(mitadX, mitadY, lines[i])
		if (cantCuadrantes1 == 2):
			j = i + 1
			while(j < lines.size()):
				cantCuadrantes2, cuadrantes2 = GetCuadrantesCubiertos(mitadX, mitadY, lines[j])
				if (cantCuadrantes2 == 2 and !CheckIfExisteEn(cuadrantes1, cuadrantes2)):
					return true
				for k in [0,1,2,3]:
					cuadrantes2[k] = false
		for k in [0,1,2,3]:
			cuadrantes1[k] = false
	
	return false

def GetHoughParameters(imgParm, imgParm1, imgParm2):
	aux1 = cv2.GaussianBlur(imgParm, Size(7, 7), 0)
	
	# Edge detection
	Mat imgHough1 = imgParm.clone()
	Mat imgHough2 = imgParm.clone()

	imgCanny = cv2.Canny(aux1, 50, 200, 3)#Canny(aux1, imgCanny, 230, 255, 3)

	#Seteo inicial de variables
	threshold = 80
	maxThreshold = 256
	auxMax = 256
	minLineLength = imgParm.cols < imgParm.rows? imgParm.cols*0.65 : imgParm.rows * 0.65	#Se asume que el bebedero ocupará un 65% del largo o ancho de la imagen
	maxLineGap = imgParm.cols < imgParm.rows ? imgParm.cols * 0.10 : imgParm.rows * 0.10
	mitadX = imgParm.cols/2, mitadY = imgParm.rows / 2

	vector<Vec2f> lines
	vector<Vec4f> linesP
	cv2.HoughLinesP(imgCanny, linesP, 1, CV_PI / 180, maxThreshold, minLineLength, maxLineGap)

	#Se halla el valor máximo de threshold que brinde suficientes resultados (al menos una linea cruzando dos cuadrantes, para ambos duos de cuadrantes)
	step = 128
	if (linesP.size() == 0):
		while (step > 10):
			if (linesP.size() < 4 or !GetIfDosLineasCubrenCadaCuadrante(linesP, mitadX, mitadY)):
				maxThreshold = maxThreshold - step
			else:
				#Posible máximo threshold
				auxMax = maxThreshold
				maxThreshold = maxThreshold + step
			step = step / 2
			if(step > 10)
				cv2.HoughLinesP(imgCanny, linesP, 1, CV_PI / 180, maxThreshold, minLineLength, maxLineGap)
	
	maxThreshold = auxMax

	#Ejecuta el analisis con las variables definitivas
	cv2.HoughLines(imgCanny, lines, 1, CV_PI / 180, maxThreshold, minLineLength, maxLineGap)
	cv2.HoughLinesP(imgCanny, linesP, 1, CV_PI / 180, maxThreshold, minLineLength, maxLineGap)

	# Draw the lines
	cv2.line(imgParm1, Point(0,0), Point(imgParm1.cols - 1, 0), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm1, Point(0, 0), Point(0,imgParm1.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm1, Point(imgParm1.cols - 1, 0), Point(imgParm1.cols - 1, imgParm1.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm1, Point(0, imgParm1.rows - 1), Point(imgParm1.cols - 1, imgParm1.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	for line in lines:
		float rho = line[0], theta = line[1]
		Point pt1, pt2
		double a = cos(theta), b = sin(theta)
		double x0 = a * rho, y0 = b * rho
		pt1.x = cvRound(x0 + 1000 * (-b))
		pt1.y = cvRound(y0 + 1000 * (a))
		pt2.x = cvRound(x0 - 1000 * (-b))
		pt2.y = cvRound(y0 - 1000 * (a))
		cv2.line(imgParm1, pt1, pt2, Scalar(255, 255, 255), anchoLinea, LINE_AA)
		cv2.line(imgHough1, pt1, pt2, Scalar(0, 0, 255), anchoLinea, LINE_AA)

	cv2.line(imgParm2, Point(0, 0), Point(imgParm2.cols - 1, 0), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm2, Point(0, 0), Point(0, imgParm2.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm2, Point(imgParm2.cols-1, 0), Point(imgParm2.cols - 1, imgParm2.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	cv2.line(imgParm2, Point(0, imgParm2.rows - 1), Point(imgParm2.cols - 1, imgParm2.rows - 1), Scalar(255, 255, 255), anchoLinea, LINE_AA)
	for line in linesP:
		cv2.line(imgParm2, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(255, 255, 255), anchoLinea, LINE_AA)
		cv2.line(imgHough2, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), anchoLinea, LINE_AA)

	aux1 = cv2.cvtColor(imgParm1, cv2.COLOR_BGR2GRAY)
	aux2 = cv2.cvtColor(imgParm2, cv2.COLOR_BGR2GRAY)
	cv2.bitwise_not(aux1, imgParm1)
	cv2.bitwise_not(aux2, imgParm2)
	
	return imgParm1, imgParm2

def GetPlantillaFromLineasRojas(imgLineasRojas):

	vector<vector<Point>> contornos
	vector<Vec4i> jerarquia

	CannyLineasRojas = cv2.Canny(imgLineasRojas, 50, 200, 3)

	cv2.findContours(CannyLineasRojas, contornos, jerarquia, RETR_LIST, CHAIN_APPROX_SIMPLE)
	vector<vector<Point>> conPoly(contornos.size())
	vector<Rect> boundRect(contornos.size())

	for i, contorno in enumerate(contornos):
		float peri = arcLength(contorno, true)

		#Esta funcion cuenta la cantidad de vertices que encuentra en cada forma
		cv2.approxPolyDP(contorno, conPoly[i], 0.02 * peri, true)
		boundRect[i] = boundingRect(conPoly[i])

	contornoMayor = getContornoMayor(conPoly)
	Point2f centroImg(imgLineasRojas.cols, imgLineasRojas.rows)

	#SE ASUME QUE EL CONTORNO DE MAYOR AREA ES EL CORRECTO
	#MAS ADELANTE PODRÍA ASEGURARSE MEJOR A PARTIR DE VERIFICAR SI TOCA EL PUNTO CENTRAL Y ATENDER SI NO LLEGA A SER EL CASO
	cv2.drawContours(plantillaBebederoReducida, contornoMayor, -1, Scalar(255, 255, 255), FILLED)

	return plantillaBebederoReducida


def getRectanguloInternoMaximo(Mat plantilla):

	plantillaCanny = cv2.Canny(plantilla, 50, 200, 3)

	cv2.findContours(plantillaCanny, contornos, jerarquia, RETR_EXTERNAL, CHAIN_APPROX_NONE)

	#Se asume que hay un solo contorno
	contorno = contornos[0]

	posiblesRectangulos = []
	int x1, x2, y1, y2
	int area

	#Se listan todos los posibles rectangulos
	for i in contorno:
		x1 = i.x
		y1 = i.y
		for j in contorno:
			x2 = j.x
			y2 = j.y
			area = abs(y2 - y1) * abs(x2 - x1)
			posiblesRectangulos.append([Point(x1,y1), Point(x2,y2), area])

	#Se ordenan los rectangulos por tamaño
	posiblesRectangulosOrdenados = sorted(posiblesRectangulos, key=lambda rect: rect[2], reverse=True)
	
	#Se elije el primero que no tenga contacto con negro
	mejorRectEncontrada = false
	rectaValida = false
	i = 0
	cantRect = posiblesRectangulos.size()
	rect = posiblesRectangulos[0]
	while (!mejorRectEncontrada and i < cantRect):
		rect = posiblesRectangulos[i]
		x1 = rect[0][0]
		y1 = rect[0][1]
		x2 = rect[1][0]
		y2 = rect[1][1]

		rectaValida = true
		xMax = max(x1, x2) - 1
		x = xMin = min(x1, x2) + 1
		yMax = max(y1, y2) - 1
		y = yMin = min(y1, y2) + 1

		while (rectaValida and x < xMax):
			if (plantilla.at<uchar>(yMin, x) == 0 or plantilla.at<uchar>(yMax, x) == 0):
				rectaValida = false
			x+= 1
		
		while (rectaValida and y < yMax):
			if (plantilla.at<uchar>(y, xMin) == 0 or plantilla.at<uchar>(y, xMax) == 0):
				rectaValida = false
			
			y+= 1
		
		if (rectaValida):
			mejorRectEncontrada = true
		else:
			i+= 1
	
	return [ posiblesRectangulos[i].p1, posiblesRectangulos[i].p2 ]

def getGaborResults(imgParm):
	#Parametros de Gabor
	theetaArr[] = [1,3]
	sigmaArr[] = [ 1,3,5 ]
	lambdaArr[] = [ CV_PI / 4 ]
	gammaArr[] = [ 0.5 ]
	ktype = CV_64F
	ksize = Size(11, 11)
	psi = 0

	#Inicializaciones extra
	imgParmGris = cv2.cvtColor(imgParm, cv2.COLOR_BGR2GRAY)

	#cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/Muestra.png", imgParm)

	result = cv2.Mat(imgParm.rows, imgParm.cols, cv2.CV_8UC1, [0])
	#Intentos
	for theeta in theetaArr:
		for sigma in sigmaArr:
			for lambda in lambdaArr:
				for gamma in gammaArr:
					kernel = cv2.getGaborKernel(ksize, sigma, theeta, lambda, gamma, psi, ktype)
					
					filter2D(imgParmGris, capsulas, CV_8UC3, kernel)
					nombre = "/home/pi/Desktop/Tesis/Resources/GaborResults/Gabor" + to_string(theeta) + "-" + to_string(sigma) + "-" + to_string(lambda) + "-" + to_string(gamma) + ".png"
					cv2.imwrite(nombre, capsulas)

					#bitwise_or(result, capsulas, result)
					result += capsulas
	cv2.imshow("result", result)
	cv2.waitKey(0)

	#cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/Resultado.png", result)

	return result

loadValores()

#Se reduce proporcionalmente la imagen para que su lado más extenso tenga 400 pixeles
ladoMasExtenso = img.cols > img.rows ? img.cols : img.rows
porcentajeReduccion = 400.0 / ladoMasExtenso

imgReducida = cv2.resize(img, null, porcentajeReduccion, porcentajeReduccion, cv2.INTER_LINEAR)
plantillaBebederoReducida = cv2.resize(plantillaBebedero, null, porcentajeReduccion, porcentajeReduccion, cv2.INTER_LINEAR)

imgHough = cv2.Mat(imgReducida.rows, imgReducida.cols, cv2.CV_8UC3, [0, 0, 0])
imgHoughP = cv2.Mat(imgReducida.rows, imgReducida.cols, cv2.CV_8UC3, [0, 0, 0])

#Se busca los parámetros más apropiados para ejecutar Hough sobre la imagen para obtener el bebedero
imgHough, imgHoughP = GetHoughParameters(imgReducida, imgHough, imgHoughP)

#Se obtiene la plantilla del bebedero a partir de la imagen creada por Hough
plantillaBebederoReducida = GetPlantillaFromLineasRojas(imgHough)

#Una vez obtenida la plantilla, se ajusta al tamaño de la imagen original
plantillaBebedero = cv2.resize(plantillaBebederoReducida, [ancho, alto])

plantillaBebedero = cv2.cvtColor(plantillaBebedero, cv2.COLOR_BGR2GRAY)
retval, plantillaBebedero = threshold(plantillaBebedero, 100, 255, cv2.THRESH_BINARY)

#Se obtiene el rectangulo más grande capaz de contenerse dentro de la forma del bebedero
rectMaxPlantilla = getRectanguloInternoMaximo(plantillaBebedero)

plantillaBebedero = cv2.rectangle(plantillaBebedero, rectMaxPlantilla[0], rectMaxPlantilla[1], 155)
cv2.imshow("rectangulo max", plantillaBebedero)
cv2.waitKey(0)

xMin = min(rectMaxPlantilla[0].x, rectMaxPlantilla[1].x)
yMin = min(rectMaxPlantilla[0].y, rectMaxPlantilla[1].y)
anchoRect = abs(rectMaxPlantilla[1].x - rectMaxPlantilla[0].x)
altoRect = abs(rectMaxPlantilla[1].y - rectMaxPlantilla[0].y)
rectMaxBebedero = img(cv::Rect(xMin, yMin, anchoRect, altoRect)).clone()
Capsulas = getGaborResults(rectMaxBebedero)
elemKerelErode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

CapsulasBlur = cv2.dilate(Capsulas, elemKerelErode)
cv2.imshow("Capsulas Dilatadas", CapsulasBlur)
cv2.waitKey(0)

cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/ResultadoDilatacion.png", CapsulasBlur)

float porcentajeCapsulas = getPorcentajeCapsulas(CapsulasBlur, 125)

cout << "Porcentaje de cápsulas:" << porcentajeCapsulas << endl

return 0
