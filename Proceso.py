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

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

SHARED_MEMORY_FORWARD_NAME = "FFT_Data"                                                             # Nombre asigando a la Shared Memory
SHARED_MEMORY_BACKWARD_NAME = "Config_Data"                                                             # Nombre asigando a la Shared Memory
WINDOW_NAME_LIST = ['flattop','blackman','hamming','hann','bartlett','parzen','bohman']

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
    '-i', '--interval', type=float, default=0,
    help='minimum time between plot updates (default: %(default)s ms)')
# parser.add_argument(
#     '-b', '--blocksize', type=int, help='block size (in samples)', default=256)
parser.add_argument(
    '-o', '--overlaping', type=int, help='block size (in samples)', default=50)
parser.add_argument(
    '-r', '--samplerate', type=float, default=44100 ,help='sampling rate of audio device')
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
    global plotdata
    global shared_array_forward
    global window_list_index
    global window_list
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
        line.set_ydata(plotdata[:, column])
    window_list_index = int(shared_array_backward[0])
    shared_array_forward[0] = np.abs(np.fft.fft(x*(window_list[window_list_index])))
    shared_array_forward[1] = np.linspace(-int(args.samplerate/(args.downsample*2)),int(args.samplerate/(args.downsample*2)),len(shared_array_forward[0]))
    
    for column, line in enumerate(lines2):
        line.set_ydata(shared_array_forward[0])
    
    # Espectrograma en la tercera columna
    # f, t_spec, Sxx = spectrogram(plotdata[:,column], fs=args.samplerate/args.downsample)
    # cax = ax3.pcolormesh(f, t_spec, Sxx.T)
    # ax3.hist2d(x=frecuencias, y=X, bins=200, norm=colors.LogNorm(clip=True))
    return [lines1, lines2]


try:
    window_list = []
    window_list_index = 0
    sh_m_f = shared_memory.SharedMemory(create=True, size=np.zeros(shape=(args.window,args.window)).nbytes, name=SHARED_MEMORY_FORWARD_NAME)        # Creamos la Shared memory con el tamaño de las muestras y con un nombre definido
    sh_m_b = shared_memory.SharedMemory(create=True, size=np.zeros(shape=(5)).nbytes, name=SHARED_MEMORY_BACKWARD_NAME)        # Creamos la Shared memory con el tamaño de las muestras y con un nombre definido
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    window_list.append(np.ones(args.window))
    for i in range(len(WINDOW_NAME_LIST)): window_list.append(signal.get_window(WINDOW_NAME_LIST[i], args.window))

    # length = int(args.window * args.samplerate / (1000 * args.downsample))
    shared_array_forward = np.ndarray((args.window,args.window), dtype=np.float64, buffer=sh_m_f.buf)
    shared_array_backward = np.ndarray((5), dtype=np.float64, buffer=sh_m_b.buf)
    shared_array_backward[0] = 0
    shared_array_backward[1] = args.downsample
    shared_array_backward[2] = args.overlaping
    shared_array_backward[3] = args.window
    shared_array_backward[4] = args.samplerate

    plotdata = np.zeros((args.window, len(args.channels)))
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    lines1 = ax1.plot(plotdata)
    # Calcular las frecuencias asociadas
    frecuencias = np.fft.fftfreq(len(plotdata), args.downsample/args.samplerate)
    lines2 = ax2.plot(frecuencias, plotdata)
    if len(args.channels) > 1:
        ax1.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax1.axis((0, len(plotdata), -1, 1))
    ax2.set_ylim((-1, 100))
    ax1.set_yticks([0])
    ax1.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    ax3.tick_params(left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback,
        blocksize=int(args.overlaping*args.window/100))
    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=False)
    with stream:
        plt.show()
    # Cerrar el objeto de memoria compartida
    sh_m_f.close()
    sh_m_b.close()
    # Borrar la memoria compartida (esto debe hacerse solo cuando ya no la uses)
    sh_m_f.unlink()
    sh_m_b.unlink()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
