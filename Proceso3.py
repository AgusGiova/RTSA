from multiprocessing import shared_memory
import numpy as np
import signal
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as animation
import matplotlib.colors as colors

# -----------------------------------------------------------------------------------------
# Funciones a utilizar
# -----------------------------------------------------------------------------------------

def handler(signum, frame):
    global sh_m3
    print('Cerrando Shared Memory del proceso 3....')
    sh_m3.close()
    print('Cerrando Proceso 3....')
    exit()

def stop(val):
    global stop_val
    stop_val = not stop_val

def inter(val):
    global txt2
    global inter_text
    global inter_text_index
    if(inter_text_index>=5):
        inter_text_index = 0
        txt2.set_val(inter_text[0])
    else:
        inter_text_index = inter_text_index + 1
        txt2.set_val(inter_text[inter_text_index])

def Wind(val):
    global config
    global wind_text
    global wind_text_index
    global txt3
    if(wind_text_index>=6):
        wind_text_index = 0
        txt3.set_val(wind_text[0])
    else:
        wind_text_index = wind_text_index + 1
        txt3.set_val(wind_text[wind_text_index])
    config[0] = wind_text_index

def init():
    ax.set_xlim([-100,20])
    plt.subplots_adjust(bottom=0.2)

def update(j):

    global data
    global FFTs
    global f
    global f_len
    global ff

    print(f)

    f_interpolation = np.linspace(0,int(f[-1]),int(f[-1])*2*inter_val[inter_text_index])                    # Definimos un eje de frecuencia para la interpolacion

    b = np.interp(f_interpolation, f[int(f_len/2):-1], data[int(f_len/2):-1])                                          # Realizamos la interpolacion

    b = list(reversed(b))

    b = 10*np.log10(np.abs(b)/0.001)                                                        # Pasamos a decibelios (dBm)

    if(len(FFTs)<10):                                                                       # Esperamos a recibir al menos 10 FFTs para realizar el historiagrama
        ax.clear()
        FFTs.append(b)
        ff.append(f_interpolation)
    elif(stop_val):

        ax.clear()

        FFTs.pop(0)
        ff.pop(0)

        FFTs.append(b)
        ff.append(f_interpolation)

        FFTs_concatenados = np.concatenate(FFTs)
        fff = np.concatenate(ff)

        H, xedges, yedges = np.histogram2d(x=fff, y=FFTs_concatenados, bins=200, normed=colors.LogNorm(clip=True),)

        ax.set_xlim(20,22050)
        ax.pcolormesh(xedges, yedges, H.T, cmap='inferno')

# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# DEFINICION DE VARIABLES Y CONSTANTES A UTILIZAR
# -----------------------------------------------------------------------------------------
SHARED_MEMORY_FORWARD_NAME = "FFT_Data"                                                                 # Nombre de la Shared Memory donde se encuentra la FFT                                        
SHARED_MEMORY_BACKWARD_NAME = "Config_Data"                                                             # Nombre de la Shared Memory para configurar la recepcion
fps = 30                                                                                    # Frames por segundo
FS = 44100                                                                                  # Frecuencia de muestreo
TIME_IN_SEC = 0.01                                                                          # Tiempo de captura
FRAMES = 4096                                                              # Duración total de la animación en segundos      
inter_val = [1,2,5,10,20,40]                                                                # Coeficiente entre la cantidad de muestras sin interpolar y la cantidad de muestras luego de interpolar                                                                                                                                            # Tasa de muestreo en Hz
FFTs = []                                                                                   # Variable para almacenar la amplitud de un conjunto de FFTs
ff = []                                                                                     # Variable para almacenar un conjunto de ejes de frecuencias
stop_val = 1
inter_text = ['x1','x2','x5','x10','x20','x40']
inter_text_index = 0
wind_text = ['Rectangular','Flattop','Hamming','Hann','Bartlett','Parzen','Bohman']
wind_text_index = 0
first=1
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# Asignación del handler para la signal CTRL+C
# -----------------------------------------------------------------------------------------
signal.signal(signal.SIGINT,handler=handler)                                                # De esta manera podemos liberar la Shared Memory correctamente
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# SHARED MEMORY
# -----------------------------------------------------------------------------------------
sh_m_b = shared_memory.SharedMemory(create=False ,name=SHARED_MEMORY_BACKWARD_NAME)                   # Conectamos a la Shared Memory
config = np.ndarray((5), dtype=np.float64, buffer=sh_m_b.buf)                             # Redefinimos el buffer de la Shared Memory para trabajar mas comodamente

sh_m_f = shared_memory.SharedMemory(create=False ,name=SHARED_MEMORY_FORWARD_NAME)                   # Conectamos a la Shared Memory
a = np.ndarray((int(config[3]),int(config[3])), dtype=np.float64, buffer=sh_m_f.buf)                             # Redefinimos el buffer de la Shared Memory para trabajar mas comodamente
data = a[0]
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# Interpolacion para aumentar los puntos sobre el historiagrama
# -----------------------------------------------------------------------------------------
f = a[1]                                                             # Definimos el eje de frecuencia
f_len = len(f)
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# Configuramos la ventana del programa
# -----------------------------------------------------------------------------------------
fig = plt.figure()                                                                          # Construimos la figura
ax = fig.add_subplot(1,1,1)                                                                         

# Boton de stop
bton_axes1 = plt.axes([0.8, 0.05, 0.1, 0.075])
bton1 = widgets.Button(bton_axes1, 'Stop', color="yellow")
bton1.on_clicked(stop)

# Boton de interpolacion
bton_axes2 = plt.axes([0.6, 0.05, 0.1, 0.075])
bton2 = widgets.Button(bton_axes2, 'Inter', color="yellow")
bton2.on_clicked(inter)

# Boton de ventaneo
bton_axes3 = plt.axes([0.4, 0.05, 0.1, 0.075])
bton3 = widgets.Button(bton_axes3, 'Window', color="yellow")
bton3.on_clicked(Wind)

# Texto
txt_axes2 = plt.axes([0.6, 0.1, 0.1, 0.075])
txt2 = widgets.TextBox(txt_axes2, '')

# Texto de ventaneo
txt_axes3 = plt.axes([0.4, 0.1, 0.1, 0.075])
txt3 = widgets.TextBox(txt_axes3, '')
# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# Iniciamos la ventana del programa
# -----------------------------------------------------------------------------------------
ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=1000/fps)
plt.show()
# -----------------------------------------------------------------------------------------