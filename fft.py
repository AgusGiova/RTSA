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
    '-i', '--interval', type=float, default=0,
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

BINS = 400
LEN_SIZE = 30
INTERPOLATION = 4       #x4

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata, i, Z, contador, tprev
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    x = plotdata[:, 0]
    for column, line in enumerate(lines1):
        line.set_ydata(x*window)
    shared_array = abs(np.fft.fft(x*window))
    
    contador = contador + 1

    t1 = time.time()
    if(t1-tprev>=1 and tprev!=0):
        print('Cantidad de FFTs por segundo = ' + str(contador) + ' FFTs/Seg')
        contador = 0
        tprev = t1
    elif(tprev==0):
        tprev = t1

    for column, line in enumerate(lines2):
        line.set_ydata(shared_array)

    # Espectrograma en la tercera columna
    Z = np.roll(Z, -1, axis=0)
    Z[-1] = shared_array[:args.window//2]
    quadmesh.set_array(Z)

    f_interpolation = np.linspace(0,int(max(frecuencias)),len(frecuencias)*INTERPOLATION)                    # Definimos un eje de frecuencia para la interpolacion

    b = np.interp(f_interpolation, frecuencias[:args.window//2], shared_array[:args.window//2])                                          # Realizamos la interpolacion

    for i in range(len(b)):
        if(b[i]==0):
            b[i]=0.00001

    b = 20*np.log10(np.abs(b)/0.001)                                                        # Pasamos a decibelios (dBm)

    FFTs.append(b)
    ff.append(f_interpolation)

    if(len(FFTs)>LEN_SIZE):

        FFTs_concatenados = np.concatenate(FFTs)
        fff = np.concatenate(ff)

        # Espectrograma en la cuarta columna
        H = histogram2d(x=fff, y=FFTs_concatenados, bins=BINS, range=((20,22000),(-5,120)))
        espectrograma.set_array(H.T)

        FFTs.pop(0)
        ff.pop(0)
    
    # return [lines1, lines2, quadmesh]
    # return quadmesh


try:
    FFTs = []
    ff = []
    sh_m = shared_memory.SharedMemory(create=True, size=np.zeros((args.window, len(args.channels))).nbytes, name=SHARED_MEMORY_NAME)        # Creamos la Shared memory con el tamaño de las muestras y con un nombre definido
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
    if args.ventaneo is None:
        window = np.ones(args.window) # Definimos la ventanaa
    else:
        window = signal.get_window(args.ventaneo, args.window)
    # length = int(args.window * args.samplerate / (1000 * args.downsample))
    shared_array = np.ndarray((args.window, len(args.channels)), buffer=sh_m.buf)  
    plotdata = np.zeros((args.window, len(args.channels)))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
    lines1 = ax1.plot(plotdata)
    # Calcular las frecuencias asociadas
    frecuencias = np.fft.fftfreq(len(plotdata), args.downsample/args.samplerate)
    lines2 = ax2.plot(frecuencias, plotdata)
    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax1.axis((0, len(plotdata), -1, 1))
    ax2.set_ylim((-1, 10))
    ax1.set_yticks([0])
    ax1.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    X, Y = np.meshgrid(frecuencias[:args.window//2], np.arange(LEN_SIZE))
    Z = np.zeros((LEN_SIZE, args.window//2))
    quadmesh = ax3.pcolormesh(X, Y, Z, vmin=0, vmax=50)

    f_interpolation = np.linspace(0,int(max(frecuencias)),len(frecuencias)*4)                    # Definimos un eje de frecuencia para la interpolacion

    b = np.interp(f_interpolation, frecuencias[:args.window//2], np.zeros(args.window//2))                                          # Realizamos la interpolacion

    for i in range(len(b)):
        if(b[i]==0):
            b[i]=0.00001

    b = 20*np.log10(np.abs(b)/0.001)                                                        # Pasamos a decibelios (dBm)

    for i in range(LEN_SIZE):

        FFTs.append(b)
        ff.append(f_interpolation)

    FFTs_concatenados = np.concatenate(FFTs)
    fff = np.concatenate(ff)

    H, xedges, yedges = np.histogram2d(x=frecuencias[:args.window//2], y=np.zeros(args.window//2), bins=BINS, range=((20,22000),(-5,120)))
    espectrograma = ax4.pcolormesh(xedges, yedges, H.T)
    # ax3.tick_params(left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback,
        blocksize=int(args.overlaping*args.window/100))
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    with stream:
        plt.show()
    # Cerrar el objeto de memoria compartida
    sh_m.close()
    # Borrar la memoria compartida (esto debe hacerse solo cuando ya no la uses)
    sh_m.unlink()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
