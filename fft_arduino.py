#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import serial
import argparse
import queue
import sys
from scipy import signal
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram
import matplotlib.colors as colors
from multiprocessing import shared_memory
import time
from fast_histogram import histogram2d
import signal as sig
import matplotlib.widgets as widgets
import threading
import cmath

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

contador = 0
tprev = 0

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=int, default=8192, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=1,
    help='minimum time between plot updates (default: %(default)s ms)')
# parser.add_argument(
#     '-b', '--blocksize', type=int, help='block size (in samples)', default=256)
parser.add_argument(
    '-o', '--overlaping', type=int, help='Overlaping of samples in %', default=99)
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-v', '--ventaneo', type=str, help='tipos de ventana hamming, flattop...')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()
FFT_to_HIST = queue.Queue()
FFT_queue = []
FFT_queue.append(queue.Queue())
FFT_queue.append(queue.Queue())  
FFT_kill = False
FFT_thead_index = 0

inter_val = [1,2,5,10,20,40]
stop_val = 1
inter_text = ['x1','x2','x5','x10','x20','x40']
inter_text_index = 0
wind_text = ['Rectangular','Flattop','Hamming','Hann','Bartlett','Parzen','Bohman']
wind_text_index = 0
first=1
window_list = []
hist_range = ((1,1000),(-100,5))
u_i = 0

"""

u_i/sqrt(n)

timepo:

matriz de incertidumbres y hacer sqrt(sumar(u_i^2))


"""

Fs = 2000

BINS = 200
LEN_SIZE = 50
WINDOW_NAME_LIST = ['flattop','blackman','hamming','hann','bartlett','parzen','bohman']

amplitud_solicitada = 0
frecuencia_solicitada = 250

ADC_UNCERTAINTY = 1   #No es porcentual (1 mV) (Valor a modo de prueba)

def handler(signum, frame):
    global FFT_kill

    FFT_kill = True

    print("Kill a todos los Threaths [" + str(FFT_kill) + "]")

    exit()

def get_uncertainty(y,x):

    u = 0
    for i in range(args.window):
        u = u + cmath.exp(((-1j)*2*np.pi*x*i)/args.window)

    u = cmath.sqrt(u)*ADC_UNCERTAINTY

    u_real = np.real(u)
    u_imag = np.imag(u)

    u_real_chi2 = 2*(u_real**4) + 4*(u_real**2)*(np.real(y[x])**2)
    u_imag_chi2 = 2*(u_imag**4) + 4*(u_imag**2)*(np.imag(y[x])**2)

    u_sum2 = u_real_chi2 + u_imag_chi2

    u_val = (1/(4*np.sqrt((u_real**2)+(u_imag**2))))*u_sum2
    u_val = np.sqrt(u_val)

    return u_val

def get_uncertainty_no_array(y,x):

    u = 0
    for i in range(args.window):
        u = u + cmath.exp(((-1j)*2*np.pi*x*i)/args.window)

    u = cmath.sqrt(u)*ADC_UNCERTAINTY

    u_real = np.real(u)
    u_imag = np.imag(u)

    u_real_chi2 = 2*(u_real**4) + 4*(u_real**2)*(np.real(y)**2)
    u_imag_chi2 = 2*(u_imag**4) + 4*(u_imag**2)*(np.imag(y)**2)

    u_sum2 = u_real_chi2 + u_imag_chi2

    u_val = (1/(4*np.sqrt((u_real**2)+(u_imag**2))))*u_sum2
    u_val = np.sqrt(u_val)

    return u_val

def get_FMBW(x,y):
    max_value = np.max(y)
    max_index_value = x[np.argmax(y)]

    x_20db_values = np.where(y<=(max_value-20))[0]

    x_max_20db_value = x_20db_values[x_20db_values>max_index_value]
    x_min_20db_value = x_20db_values[x_20db_values<max_index_value]

    x_max_20db_value = x_max_20db_value[np.abs(x_max_20db_value-max_index_value).argmin()] if x_max_20db_value.size > 0 else None
    x_min_20db_value = x_min_20db_value[np.abs(x_min_20db_value-max_index_value).argmin()] if x_min_20db_value.size > 0 else None

    return [max_index_value, max_value, x_min_20db_value, x_max_20db_value]

def get_hist2d_curve(hist, xedges, yedges):
    # Encontrar el índice del primer valor no cero desde arriba hacia abajo en cada columna
    first_nonzero_indices = np.argmax(hist[::-1, :] > 0, axis=0)

    # Filtrar columnas donde no hay valores no cero
    valid_columns = np.any(hist > 0, axis=0)
    first_nonzero_indices = first_nonzero_indices[valid_columns]
    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
    x_bin_centers = x_bin_centers[valid_columns]

    # Convertir los índices de bins en valores Y
    first_y_values = yedges[-1 - first_nonzero_indices]

    return [x_bin_centers, first_y_values]


def stop(val):
    global stop_val
    stop_val = not stop_val

def inter(val):
    global txt2
    global amplitud_solicitada
    global frecuencia_solicitada
    
    datos = txt2.text.split(sep=',')

    amplitud_solicitada = datos[0]
    frecuencia_solicitada = datos[1]

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

def init():
    plt.subplots_adjust(bottom=0.2)

"""def FFT1_callback(data):
    global window_list
    global wind_text_index
    global FFT_kill
    global FFT_queue
    global FFTs
    global ff

    global contador

    # Configuración de pyFFTW
    fft_size = args.window
    pyfftw.config.NUM_THREADS = 4  # Ajusta el número de hilos según tu sistema
    fft_buffer = pyfftw.empty_aligned(fft_size, dtype='complex128')
    fft_result = pyfftw.empty_aligned(fft_size, dtype='complex128')
    fft_plan = pyfftw.FFTW(fft_buffer, fft_result, direction='FFTW_FORWARD')

    while not FFT_kill:
        try:
            # Espera datos en la cola
            data = FFT_queue[0].get()

            # Multiplica por la ventana correspondiente
            data_windowed = data * window_list[wind_text_index]

            # Copia los datos al búfer de pyFFTW
            np.copyto(fft_buffer, data_windowed)

            # Ejecuta la FFT con pyFFTW
            fft_plan.execute()

            # Obtiene la magnitud en decibelios
            b = np.abs(fft_result)
            b[b == 0] = 0.00001  # Evitar valores cero
            b = 20 * np.log10(b / len(b))

            # Pasa la mitad de la FFT a la cola para el histograma
            FFT_to_HIST.put(b[:args.window // 2])
            contador += 1
        except Exception as e:
            print(f"Error en FFT1_callback: {e}")"""


def FFT1_callback(data):
    global window_list
    global wind_text_index
    global FFT_kill
    global FFT_queue

    global contador

    while (FFT_kill==False):

        data = FFT_queue[0].get()

        contador = contador + 1

        fft_result = np.fft.fft(data*(window_list[wind_text_index]))
        

        #print(get_uncertainty(fft_result,100)/np.abs(fft_result[100]))

        b = np.abs(fft_result)

        for i in range(len(b)):
            if(b[i]==0):
                b[i]=0.00001

        b = 20*np.log10(b/len(b))                                                        

        FFT_to_HIST.put(b[:args.window//2])  

"""def audio_callback(indata, frames, time, status):
    This is called (from a separate thread) for each audio block.
    global plotdata, FFT_queue, FFT_thead_index

    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    shift = len(indata[::args.downsample, mapping])
    plotdata = np.roll(plotdata, -shift, axis=0)

    plotdata[-shift:, :] = indata[::args.downsample, mapping]

    FFT_queue[FFT_thead_index].put(plotdata[:, 0])"""

PUERTO = "/dev/ttyUSB0"
baudrate = 500000
shift = 0

def sample_callback(data):
    global FFT_kill
    global plotdata
    shift = 0

    try:
        ser = serial.Serial(port=PUERTO, baudrate=baudrate)
    except serial.SerialException as e:
        print(f"Error abriendo el puerto: {e}")
        FFT_kill = True
    
    contador = 0
    t1 = time.time()
    taux = time.time()

    while(FFT_kill==False):

        if(ser.in_waiting>0):
            linea = ser.readline().decode("latin-1").strip()

            if linea.isdecimal():
                numero = int(linea)
                contador = contador + 1
                plotdata = np.roll(plotdata, -1)
                plotdata[-1] = numero*5/1023
                shift = shift + 1

            if((shift/len(plotdata))>=(1-(args.overlaping/100))):
                FFT_queue[FFT_thead_index].put(plotdata)
                shift = 0
            
            """t1 = time.time()
            if(t1-taux>=1):
                print("Fs = " + str(contador))
                contador = 0
                taux = t1"""

def HISTOGRAM_callback(data):
    global FFT_kill
    global i, Z, contador, tprev, window_list, wind_text_index, FFTs, ff, FFT_to_HIST

    FFTs = np.zeros(args.window*LEN_SIZE)
    ff = np.zeros(args.window*LEN_SIZE)

    shift = int(((LEN_SIZE-1)*args.window) + (args.window/2) - 1)

    while (FFT_kill==False):

        """t1 = time.time()
        if((t1-tprev)>=1 and tprev!=0):
            print('Cantidad de FFTs por segundo = ' + str(contador) + ' FFTs/Seg')
            contador = 0
            tprev = t1
        elif(tprev==0):
            tprev = t1"""

        aux = FFT_to_HIST.get()

        FFTs = np.roll(FFTs,-args.window)

        FFTs[shift:-1] = aux

        ff = np.roll(ff,-args.window)
        ff[shift:-1] = frecuencias[:args.window//2]

        H = histogram2d(x=ff, y=FFTs, bins=BINS, range=hist_range)
        q.put(H)


def update_plot(frame):
    global plotdata, Z, window_list, wind_text_index, FFTs, stop_val, amplitud_solicitada, frecuencia_solicitada

    xedges = np.linspace(hist_range[0][0],hist_range[0][1],BINS+1)
    yedges = np.linspace(hist_range[1][0],hist_range[1][1],BINS+1)

    lines1.set_ydata(plotdata*(window_list[wind_text_index]))

    # Espectrograma en la tercera columna
    if(len(FFTs)==len(Z)):
        Z = np.roll(Z, -1, axis=0)
        Z[-1] = FFTs[-1]
        quadmesh.set_array(Z)

    if(q.empty() == False):
        H = q.get()

        H = H.T

        x_bin_centers, first_y_values = get_hist2d_curve(H,xedges,yedges)

        if(len(first_y_values)>0 and len(x_bin_centers)>0):
            ploteo.set_xdata(x_bin_centers)
            ploteo.set_ydata(first_y_values)

        if(stop_val==1):
            x_max_value, y_max_value, f1, f2 = get_FMBW(x_bin_centers,first_y_values)

            """print("--------------------------------------")
            print(f"Valor maximo en frecuencia = {x_max_value}")
            print(f"Valor maximo en dB = {y_max_value}")
            print(f"Valor f1 = {f1}")
            print(f"Valor f2 = {f2}")
            print("--------------------------------------")"""

            amplitud_solicitada = int(amplitud_solicitada)
            frecuencia_solicitada = int(frecuencia_solicitada)

            array = np.asarray(xedges) 
            idx_f = (np.abs(array - frecuencia_solicitada)).argmin()
            frecuencia_corregida = xedges[idx_f]

            array = np.asarray(yedges) 
            idx_a = (np.abs(array - amplitud_solicitada)).argmin()
            amplitud_corregida = yedges[idx_a]

            u_i = get_uncertainty_no_array(amplitud_corregida,frecuencia_corregida)

            print("---")
            print(frecuencia_corregida)
            print(idx_f)
            print(amplitud_corregida)
            print(idx_a)
            print("---")

            u_i_informada = u_i/np.sqrt(H.T[idx_f,idx_a])

            print(u_i_informada)

            espectrograma.set_array(H)
    


try:
    sig.signal(sig.SIGINT,handler=handler) 
    FFTs = []
    ff = []
    plotdata = np.zeros(args.window)
    frecuencias = np.fft.fftfreq(len(plotdata), args.downsample/Fs)        # Fs = 500Hz

    FFT1 = threading.Thread(target=FFT1_callback, args=(1,), daemon=True)
    FFT1.start()
    HISTOGRAM = threading.Thread(target=HISTOGRAM_callback, args=(1,), daemon=True)
    HISTOGRAM.start()
    samples = threading.Thread(target=sample_callback, args=(1,), daemon=True)
    samples.start()
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    window_list.append(np.ones(args.window))
    for i in range(len(WINDOW_NAME_LIST)): window_list.append(signal.get_window(WINDOW_NAME_LIST[i], args.window))

    if args.ventaneo is None:
        wind_text_index = 0 # Definimos la ventanaa
    else:
        for i in range(len(window_list)):
            if(args.ventaneo==WINDOW_NAME_LIST[i]): wind_text_index = i
    # length = int(args.window * args.samplerate / (1000 * args.downsample))
    fig, (ax1, ax3, ax4) = plt.subplots(3,1)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(bottom=0.25)
    lines1, = ax1.plot(plotdata)
    # Calcular las frecuencias asociadas
    
    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax1.axis((0, len(plotdata), -1, 1))
    ax1.set_yticks([0])
    #ax1.tick_params(bottom=False, top=False, labelbottom=False,
    #               right=False, left=False, labelleft=False)
    X, Y = np.meshgrid(frecuencias[:args.window//2], np.arange(LEN_SIZE))
    Z = np.zeros((LEN_SIZE, args.window//2))
    quadmesh = ax3.pcolormesh(X, Y, Z, vmin=0, vmax=50)

    FFTs = np.zeros(args.window*LEN_SIZE)
    ff = np.zeros(args.window*LEN_SIZE)

    H, xedges, yedges = np.histogram2d(x=frecuencias[:args.window//2], y=np.zeros(args.window//2), bins=BINS, range=hist_range)
    espectrograma = ax4.pcolormesh(xedges, yedges, H.T)

    first_y_values = np.zeros(len(xedges))

    ploteo, = ax4.plot(xedges, first_y_values, linestyle='-', color="coral")

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

    #stream = sd.InputStream(
    #    device=args.device, channels=max(args.channels),
    #    samplerate=args.samplerate, callback=audio_callback,
    #    blocksize=int(args.overlaping*args.window/100))
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    #with stream:
    #    plt.show()
    plt.show()
    FFT_kill = True
except Exception as e:
    parser.exit("Error en la inicializacion-" + type(e).__name__ + ': ' + str(e))
