import cv2
import numpy as np
import math

#Asignación de variables globales
path = "/home/pi/Desktop/Tesis/Resources/Prueba1.jpg" #"/home/pi/Desktop/Tesis/Resources/Vid_10_1.jpg"#'/home/pi/Desktop/Tesis/Resources/CajaBlanca.jpeg'#"/home/pi/Desktop/Tesis/Resources/Prueba7.jpg" #

global alto
alto = 0
global ancho
ancho = 0
global anchoLinea
anchoLinea = 3

def loadValores():
	global img
	img= cv2.imread(path)
	global imgTest
	imgTest = cv2.imread(path)
	global alto
	global ancho
	alto, ancho, _ = img.shape
	global plantillaBebedero
	plantillaBebedero = np.zeros((alto, ancho, 3), dtype = "uint8")

def dentroBebedero(plantilla, x, y):
	return (plantilla[y,x] > 0)


def getPorcentajeCapsulas(imgParm, umbral):
	altoParm,anchoParm = imgParm.shape
	totalPixeles = anchoParm*altoParm
	pixelesCapsulas = 0
	print('Entro a for de calculo de porcentaje')
	
	for row in imgParm:
		for pixel in row:
			if (pixel >= umbral):
				pixelesCapsulas += 1
	
	print('Salio de for de calculo de porcentaje')
	
	return pixelesCapsulas / totalPixeles


def getContornoMayor(contornos):
	contornoMayor = 0
	areaMayor = cv2.contourArea(contornos[0])
	for contorno in contornos:
		nuevaArea = cv2.contourArea(contorno)
		if (nuevaArea > areaMayor):
			areaMayor = nuevaArea
			contornoMayor = contorno
	
	return contornoMayor

def GetCuadrantesCubiertos(mitadX, mitadY, line):
	cuadrantes = [False, False, False, False]
	if (line[0] < mitadX):
		if (line[1] < mitadY):
			cuadrantes[1] = True
		else:
			cuadrantes[2] = True
	else:
		if (line[1] < mitadY):
			cuadrantes[0] = True
		else:
			cuadrantes[3] = True
	if (line[2] < mitadX):
		if (line[3] < mitadY):
			cuadrantes[1] = True
		else:
			cuadrantes[2] = True
	else:
		if (line[3] < mitadY):
			cuadrantes[0] = True
		else:
			cuadrantes[3] = True
	count = 0
	for cuadrante in cuadrantes:
		if cuadrante:
			count+=1
	
	return count, cuadrantes

def CheckIfExisteEn(arr, valoresARevisar):
	for i, arrVal in enumerate(arr):
		if (arrVal == valoresARevisar[i]):
			return True
	
	return False

def GetIfDosLineasCubrenCadaCuadrante(lines, mitadX, mitadY):
	cuadrantes1 = [ False, False, False, False ]
	cuadrantes2 = [ False, False, False, False ]
	
	i = 0
	while(i < lines.__len__()):
		cantCuadrantes1, cuadrantes1 = GetCuadrantesCubiertos(mitadX, mitadY, lines[i][0])
		if (cantCuadrantes1 == 2):
			j = i + 1
			while(j < lines.__len__()):
				cantCuadrantes2, cuadrantes2 = GetCuadrantesCubiertos(mitadX, mitadY, lines[j][0])
				if (cantCuadrantes2 == 2 and not CheckIfExisteEn(cuadrantes1, cuadrantes2)):
					return True
				for k in [0,1,2,3]:
					cuadrantes2[k] = False
				j = j + 1
		for k in [0,1,2,3]:
			cuadrantes1[k] = False
		
		i = i + 1
	
	return False

def GetHoughParameters(imgParm, imgParm1, imgParm2):
	aux1 = cv2.GaussianBlur(imgParm, (7, 7), 0)
	
	# Edge detection
	imgHough1 = np.copy(imgParm)#imgParm.clone()
	imgHough2 = np.copy(imgParm)#imgParm.clone()

	imgCanny = cv2.Canny(aux1, 50, 200, 3)#Canny(aux1, imgCanny, 230, 255, 3)
	
	#Seteo inicial de variables
	threshold = 80
	maxThreshold = 256
	auxMax = 256
	imgParmAlto,imgParmAncho,_ = imgParm.shape
	if(imgParmAncho < imgParmAlto):	#Se asume que el bebedero ocupará un 65% del largo o ancho de la imagen
		minLineLength = imgParmAncho*0.65
	else:
		minLineLength = imgParmAlto * 0.65
	if(imgParmAncho < imgParmAlto):
		maxLineGap = imgParmAncho * 0.10
	else:
		maxLineGap = imgParmAlto * 0.10
	mitadX = (imgParm.shape)[1] / 2
	mitadY = imgParmAlto / 2

	linesP = cv2.HoughLinesP(imgCanny, 1, math.pi / 180, maxThreshold, None, minLineLength, maxLineGap)
	
	#Se halla el valor máximo de threshold que brinde suficientes resultados (al menos una linea cruzando dos cuadrantes, para ambos duos de cuadrantes)
	step = 128
	while (step > 10):
		if (linesP is None):
			maxThreshold = int(maxThreshold - step)
		else:
			if((linesP.__len__() < 4) or (not (GetIfDosLineasCubrenCadaCuadrante(linesP, mitadX, mitadY)))):
				maxThreshold = int(maxThreshold - step)
			else:
				#Posible máximo threshold
				auxMax = maxThreshold
				print(str(auxMax))
				maxThreshold = int(maxThreshold + step)
		step = step / 2
		if(step > 10):
			linesP = cv2.HoughLinesP(imgCanny, 1, math.pi / 180, maxThreshold, None, minLineLength, maxLineGap)
	
	maxThreshold = auxMax

	#Ejecuta el analisis con las variables definitivas
	lines = cv2.HoughLines(imgCanny, 1, math.pi / 180, maxThreshold, None, minLineLength, maxLineGap)
	linesP = cv2.HoughLinesP(imgCanny, 1, math.pi / 180, maxThreshold, None, minLineLength, maxLineGap)

	# Draw the lines
	imgParm1Alto,imgParm1Ancho,_ = imgParm1.shape
	imgParm1 = cv2.line(imgParm1, (0,0), (imgParm1Ancho - 1, 0), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	imgParm1 = cv2.line(imgParm1, (0, 0), (0,imgParm1Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	imgParm1 = cv2.line(imgParm1, (imgParm1Ancho- 1, 0), (imgParm1Ancho - 1, imgParm1Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	imgParm1 = cv2.line(imgParm1, (0, imgParm1Alto - 1), (imgParm1Ancho - 1, imgParm1Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	for line in lines:
		rho = line[0][0]
		theta = line[0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = a * rho
		y0 = b * rho
		pt1 = (round(x0 + 1000 * (-b)), round(y0 + 1000 * (a)))
		pt2 = (round(x0 - 1000 * (-b)), round(y0 - 1000 * (a)))
		cv2.line(imgParm1, pt1, pt2, (255, 255, 255), anchoLinea, cv2.LINE_AA)
		cv2.line(imgHough1, pt1, pt2, (0, 0, 255), anchoLinea, cv2.LINE_AA)

	imgParm2Alto,imgParm2Ancho, _ = imgParm2.shape
	cv2.line(imgParm2, (0, 0), (imgParm2Ancho - 1, 0), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	cv2.line(imgParm2, (0, 0), (0, imgParm2Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	cv2.line(imgParm2, (imgParm2Ancho - 1, 0), (imgParm2Ancho - 1, imgParm2Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	cv2.line(imgParm2, (0, imgParm2Alto - 1), (imgParm2Ancho - 1, imgParm2Alto - 1), (255, 255, 255), anchoLinea, cv2.LINE_AA)
	for line in linesP:
		cv2.line(imgParm2, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), anchoLinea, cv2.LINE_AA)
		cv2.line(imgHough2, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), anchoLinea, cv2.LINE_AA)

	aux1 = cv2.normalize(imgParm1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	aux2 = cv2.normalize(imgParm2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	imgParm1 = cv2.cvtColor(aux1, cv2.COLOR_BGRA2GRAY)
	imgParm2 = cv2.cvtColor(aux2, cv2.COLOR_BGRA2GRAY)
	
	cv2.bitwise_not(imgParm1, aux1)
	cv2.bitwise_not(imgParm2, aux2)
	
	return aux1, aux2

def GetPlantillaFromLineasRojas(imgLineasRojas):
	CannyLineasRojas = cv2.Canny(imgLineasRojas, 50, 200, 3)

	contornos, jerarquia = cv2.findContours(CannyLineasRojas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	conPoly = []
	boundRect = []

	for i, contorno in enumerate(contornos):
		peri = cv2.arcLength(contorno, True)

		#Esta funcion cuenta la cantidad de vertices que encuentra en cada forma
		conPoly.append(cv2.approxPolyDP(contorno, 0.02 * peri, False))
		boundRect.append(cv2.boundingRect(conPoly[i]))

	print(str(boundRect[0]))
	
	contornoMayor = getContornoMayor(conPoly)

	#SE ASUME QUE EL CONTORNO DE MAYOR AREA ES EL CORRECTO
	#MAS ADELANTE PODRÍA ASEGURARSE MEJOR A PARTIR DE VERIFICAR SI TOCA EL PUNTO CENTRAL Y ATENDER SI NO LLEGA A SER EL CASO
	cv2.drawContours(plantillaBebederoReducida, [contornoMayor], -1, (255, 255, 255), cv2.FILLED)

	return plantillaBebederoReducida


def getRectanguloInternoMaximo(plantilla):

	plantillaCanny = cv2.Canny(plantilla, 50, 200, 3)

	contornos, jerarquia = cv2.findContours(plantillaCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	#Se asume que hay un solo contorno
	contorno = contornos[0]

	#print(contorno)

	posiblesRectangulos = []
	
	#El rectangulo a hallar debe contener un tamaño mínimo asumiendo que se enfocó bien al bebedero, para reducir resultados a buscar
	areaMinAceptable = ladoMasExtenso * 0.4
	
	print('entro a for')

	#Se listan todos los posibles rectangulos
	for i,puntos in enumerate(contorno):
		x1 = puntos[0][0]
		y1 = puntos[0][1]
		j = i + 1
		while (j < contorno.__len__()):
			x2 = contorno[j][0][0]
			y2 = contorno[j][0][1]
			area = abs(y2 - y1) * abs(x2 - x1)
			if(area > areaMinAceptable):
				posiblesRectangulos.append([[x1,y1], [x2,y2], area])
			j = j + 1

	print('salio de for')

	#Se ordenan los rectangulos por tamaño
	posiblesRectangulosOrdenados = sorted(posiblesRectangulos, key=lambda rect: rect[2], reverse=True)
	
	print('termino de ordenar')
	
	#Se elije el primero que no tenga contacto con negro
	mejorRectEncontrada = False
	rectaValida = False
	i = 0
	cantRect = len(posiblesRectangulosOrdenados)
	rect = posiblesRectangulosOrdenados[0]
	while (not mejorRectEncontrada and i < cantRect):
		rect = posiblesRectangulosOrdenados[i]
		
		x1 = rect[0][0]
		y1 = rect[0][1]
		x2 = rect[1][0]
		y2 = rect[1][1]

		rectaValida = True
		xMax = max(x1, x2) - 1
		x = xMin = min(x1, x2) + 1
		yMax = max(y1, y2) - 1
		y = yMin = min(y1, y2) + 1

		while (rectaValida and x < xMax):
			if (plantilla[yMin, x] == 0 or plantilla[yMax, x] == 0):
				rectaValida = False
			x+= 1
		
		while (rectaValida and y < yMax):
			if (plantilla[y, xMin] == 0 or plantilla[y, xMax] == 0):
				rectaValida = False
			
			y+= 1
		
		if (rectaValida):
			mejorRectEncontrada = True
		else:
			i+= 1
	
	print('encontro el mejor')
	
	return [ posiblesRectangulosOrdenados[i][0], posiblesRectangulosOrdenados[i][1] ]

def getGaborResults(imgParm):
	#Parametros de Gabor
	theetaArr = [1,3]
	sigmaArr = [ 1,3,5 ]
	lambdaArr = [ math.pi / 4 ]
	gammaArr = [ 0.5 ]
	ktype = cv2.CV_64F
	ksize = (11, 11)
	psi = 0

	#Inicializaciones extra
	imgParmGris = cv2.cvtColor(imgParm, cv2.COLOR_BGR2GRAY)

	#cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/Muestra.png", imgParm)
	imgParmAlto, imgParmAncho, _ = imgParm.shape

	result = np.zeros(imgParmGris.shape, dtype = "uint8")#cv2.Mat(imgParmAlto, imgParmAncho, cv2.CV_8UC1, [0])
	#Intentos
	for theeta in theetaArr:
		for sigma in sigmaArr:
			for lambdaVal in lambdaArr:
				for gamma in gammaArr:
					kernel = cv2.getGaborKernel(ksize, sigma, theeta, lambdaVal, gamma, psi, ktype)
					
					capsulas = cv2.filter2D(imgParmGris, cv2.CV_8UC3, kernel)
					nombre = "/home/pi/Desktop/Tesis/Resources/GaborResults/Gabor" + str(theeta) + "-" + str(sigma) + "-" + str(lambdaVal) + "-" + str(gamma) + ".png"
					cv2.imwrite(nombre, capsulas)

					#cv2.bitwise_or(result, capsulas, result)
					result = result + capsulas
	cv2.imshow("result", result)
	cv2.waitKey(0)

	#cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/Resultado.png", result)

	return result

loadValores()

#Se reduce proporcionalmente la imagen para que su lado más extenso tenga 400 pixeles
global ladoMasExtenso

if(ancho > alto):
	ladoMasExtenso = ancho
	anchoNuevo = 400
	altoNuevo = 0
else:
	ladoMasExtenso = alto
	altoNuevo = 400
	anchoNuevo = 0

porcentajeReduccion = 400.0 / ladoMasExtenso

if(altoNuevo == 0):
	altoNuevo = math.floor(alto * porcentajeReduccion)
else:
	anchoNuevo = math.floor(ancho * porcentajeReduccion)

imgReducida = cv2.resize(img, (anchoNuevo,altoNuevo), cv2.INTER_LINEAR)
plantillaBebederoReducida = cv2.resize(plantillaBebedero, (anchoNuevo,altoNuevo), cv2.INTER_LINEAR)

cv2.imshow('imgReducida',imgReducida)
cv2.waitKey(0)

imgHough = np.zeros((altoNuevo,anchoNuevo,3))	#cv2.CreateMat(altoNuevo, anchoNuevo, cv2.CV_8UC3, [0, 0, 0]) 
imgHoughP = np.zeros((altoNuevo,anchoNuevo,3))	#cv2.CreateMat(altoNuevo, anchoNuevo, cv2.CV_8UC3, [0, 0, 0])

#Se busca los parámetros más apropiados para ejecutar Hough sobre la imagen para obtener el bebedero
imgHough, imgHoughP = GetHoughParameters(imgReducida, imgHough, imgHoughP)

cv2.imshow('imgHough',imgHough)
cv2.imshow('imgHoughP',imgHoughP)
cv2.waitKey(0)

#Se obtiene la plantilla del bebedero a partir de la imagen creada por Hough
plantillaBebederoReducida = GetPlantillaFromLineasRojas(imgHough)

cv2.imshow('plantillaBebederoReducida',plantillaBebederoReducida)
cv2.waitKey(0)

plantillaBebederoReducida = cv2.cvtColor(plantillaBebederoReducida, cv2.COLOR_BGR2GRAY)
retval, plantillaBebederoReducida = cv2.threshold(plantillaBebederoReducida, 100, 255, cv2.THRESH_BINARY)

#Se obtiene el rectangulo más grande capaz de contenerse dentro de la forma del bebedero
rectMaxPlantillaReducida = getRectanguloInternoMaximo(plantillaBebederoReducida)

rectMaxPlantillaReducida[0][0] = int(rectMaxPlantillaReducida[0][0])
rectMaxPlantillaReducida[0][1] = int(rectMaxPlantillaReducida[0][1])
rectMaxPlantillaReducida[1][0] = int(rectMaxPlantillaReducida[1][0])
rectMaxPlantillaReducida[1][1] = int(rectMaxPlantillaReducida[1][1])

plantillaBebederoReducida = cv2.rectangle(plantillaBebederoReducida, rectMaxPlantillaReducida[0], rectMaxPlantillaReducida[1], (155), -1)
cv2.imshow("rectangulo max", plantillaBebederoReducida)
cv2.waitKey(0)

xMin = min(rectMaxPlantillaReducida[0][0], rectMaxPlantillaReducida[1][0])
yMin = min(rectMaxPlantillaReducida[0][1], rectMaxPlantillaReducida[1][1])
xMax = max(rectMaxPlantillaReducida[0][0], rectMaxPlantillaReducida[1][0])
yMax = max(rectMaxPlantillaReducida[0][1], rectMaxPlantillaReducida[1][1])

print(xMin)
print(yMin)
print(xMax)
print(yMax)

xMin = int(xMin/porcentajeReduccion)
xMax = int(xMax/porcentajeReduccion)
yMin = int(yMin/porcentajeReduccion)
yMax = int(yMax/porcentajeReduccion)

rectMaxBebedero = img[yMin:yMax,xMin:xMax]

cv2.imshow('rectMaxBebedero',rectMaxBebedero)
cv2.waitKey(0)

Capsulas = getGaborResults(rectMaxBebedero)
elemKerelErode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

CapsulasBlur = cv2.dilate(Capsulas, elemKerelErode)
cv2.imshow("Capsulas Dilatadas", CapsulasBlur)
cv2.waitKey(0)

cv2.imwrite("/home/pi/Desktop/Tesis/Resources/GaborResults/ResultadoDilatacion.png", CapsulasBlur)

porcentajeCapsulas = getPorcentajeCapsulas(CapsulasBlur, 125)

print("Porcentaje de cápsulas:" + str(porcentajeCapsulas))
