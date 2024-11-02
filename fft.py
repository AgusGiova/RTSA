#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
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

SHARED_MEMORY_NAME = "Data"                                                             # Nombre asigando a la Shared Memory
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
    '-w', '--window', type=int, default=1024, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=1,
    help='minimum time between plot updates (default: %(default)s ms)')
# parser.add_argument(
#     '-b', '--blocksize', type=int, help='block size (in samples)', default=256)
parser.add_argument(
    '-o', '--overlaping', type=int, help='block size (in samples)', default=50)
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

BINS = 400
LEN_SIZE = 100
INTERPOLATION = 4       #x4
WINDOW_NAME_LIST = ['flattop','blackman','hamming','hann','bartlett','parzen','bohman']

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


def get_hist2d_curve(hist,xedges,yedges):

    first_y_values = []
    x_bin_centers = []

    # Recorrer cada columna del histograma (eje X)
    for i in range(hist.shape[1]-1):
        # Recorrer los bins del eje Y desde arriba hacia abajo
        for j in range(hist.shape[0]-1):
            if hist[i, (hist.shape[0]-1)-j] > 0:  # Si el valor es distinto de cero
                x_bin_centers.append((xedges[i] + xedges[i+1]) / 2)  # Centro del bin en X
                first_y_values.append(yedges[(hist.shape[0]-1)-j])  # Valor del bin en Y
                break

    first_y_values = np.atleast_1d(first_y_values)
    x_bin_centers = np.atleast_1d(x_bin_centers)

    return [x_bin_centers, first_y_values]

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

def init():
    plt.subplots_adjust(bottom=0.2)

def FFT1_callback(data):
    global window_list
    global wind_text_index
    global FFT_kill
    global FFT_queue
    global FFTs
    global ff

    global contador
    
    print("En el thread 1")

    while (FFT_kill==False):

        data = FFT_queue[0].get()

        contador = contador + 1

        fft_result = np.fft.fft(data*(window_list[wind_text_index]))

        print(get_uncertainty(fft_result,1000)/np.abs(fft_result[1000]))

        b = np.abs(fft_result)

        for i in range(len(b)):
            if(b[i]==0):
                b[i]=0.00001

        b = 20*np.log10(b/len(b))                                                        

        FFT_to_HIST.put(b[:args.window//2])   

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    global plotdata, FFT_queue, FFT_thead_index

    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    shift = len(indata[::args.downsample, mapping])
    plotdata = np.roll(plotdata, -shift, axis=0)

    plotdata[-shift:, :] = indata[::args.downsample, mapping]

    FFT_queue[FFT_thead_index].put(plotdata[:, 0])

def HISTOGRAM_callback(data):
    global FFT_kill
    global plotdata, i, Z, contador, tprev, window_list, wind_text_index, FFTs, ff, FFT_to_HIST

    while (FFT_kill==False):

        t1 = time.time()
        if((t1-tprev)>=1 and tprev!=0):
            print('Cantidad de FFTs por segundo = ' + str(contador) + ' FFTs/Seg')
            contador = 0
            tprev = t1
        elif(tprev==0):
            tprev = t1

        aux = FFT_to_HIST.get()

        FFTs.append(aux)
        ff.append(frecuencias[:args.window//2])

        if(len(FFTs)>3 and len(ff)>3):
            FFTs_concatenados = np.concatenate(FFTs)
            fff = np.concatenate(ff)

            if(len(FFTs)>LEN_SIZE and len(FFTs_concatenados)==len(fff)):

                # Espectrograma en la cuarta columna
                H = histogram2d(x=fff, y=FFTs_concatenados, bins=BINS, range=((20,22000),(-120,20)))
                q.put(H)

                FFTs = FFTs[:len(FFTs)//2]
                ff = ff[:len(ff)//2]

def update_plot(frame):
    global plotdata, Z, window_list, wind_text_index, FFTs, stop_val

    xedges = np.linspace(20,22000,BINS)
    yedges = np.linspace(-120,20,BINS)

    lines1.set_ydata(plotdata[:, 0]*(window_list[wind_text_index]))

    # Espectrograma en la tercera columna
    if(len(FFTs[-1])==len(Z[-1])):
        Z = np.roll(Z, -1, axis=0)
        Z[-1] = FFTs[-1]
        quadmesh.set_array(Z)

    if(q.empty() == False):
        H = q.get()

        x_bin_centers, first_y_values = get_hist2d_curve(H,xedges,yedges)

        if(len(first_y_values)>0 and len(x_bin_centers)>0):
            ploteo.set_xdata(x_bin_centers)
            ploteo.set_ydata(first_y_values)

        if(stop_val==1):
            espectrograma.set_array(H.T)
    


try:
    sig.signal(sig.SIGINT,handler=handler) 
    FFTs = []
    ff = []
    FFT1 = threading.Thread(target=FFT1_callback, args=(1,))
    FFT1.start()
    HISTOGRAM = threading.Thread(target=HISTOGRAM_callback, args=(1,))
    HISTOGRAM.start()
    sh_m = shared_memory.SharedMemory(create=True, size=np.zeros((args.window, len(args.channels))).nbytes, name=SHARED_MEMORY_NAME)        # Creamos la Shared memory con el tamaño de las muestras y con un nombre definido
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
    shared_array = np.ndarray((args.window,), buffer=sh_m.buf)  
    plotdata = np.zeros((args.window, len(args.channels)))
    fig, (ax1, ax3, ax4) = plt.subplots(3,1)
    fig.tight_layout(pad=1)
    fig.subplots_adjust(bottom=0.25)
    lines1, = ax1.plot(plotdata)
    # Calcular las frecuencias asociadas
    frecuencias = np.fft.fftfreq(len(plotdata), args.downsample/args.samplerate)
    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax1.axis((0, len(plotdata), -1, 1))
    ax1.set_yticks([0])
    ax1.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    X, Y = np.meshgrid(frecuencias[:args.window//2], np.arange(LEN_SIZE))
    Z = np.zeros((LEN_SIZE, args.window//2))
    quadmesh = ax3.pcolormesh(X, Y, Z, vmin=0, vmax=50)

    f_interpolation = np.linspace(0,int(max(frecuencias)),len(frecuencias))                    # Definimos un eje de frecuencia para la interpolacion

    b = np.interp(f_interpolation, frecuencias[:args.window//2], np.zeros(args.window//2))                                          # Realizamos la interpolacion

    for i in range(len(b)):
        if(b[i]==0):
            b[i]=0.00001

    b = 20*np.log10(np.abs(b))                                                        # Pasamos a decibelios (dBm)

    for i in range(LEN_SIZE):

        FFTs.append(b)
        ff.append(f_interpolation)

    FFTs_concatenados = np.concatenate(FFTs)
    fff = np.concatenate(ff)

    H, xedges, yedges = np.histogram2d(x=frecuencias[:args.window//2], y=np.zeros(args.window//2), bins=BINS, range=((20,22000),(-120,20)))
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

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback,
        blocksize=int(args.overlaping*args.window/100))
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    with stream:
        plt.show()
    FFT_kill = True
    # Cerrar el objeto de memoria compartida
    sh_m.close()
    # Borrar la memoria compartida (esto debe hacerse solo cuando ya no la uses)
    sh_m.unlink()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
