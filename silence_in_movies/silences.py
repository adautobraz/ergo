import subprocess as sp
import sys
import numpy

FFMPEG_BIN = "/Users/adautobrazdasilvaneto/anaconda3/bin/ffmpeg"

print('ASplit.py <src.mp3> <silence duration in seconds> <threshold amplitude 0.0 .. 1.0>')

src = sys.argv[1]
dur = float(sys.argv[2])
thr = int(float(sys.argv[3]) * 65535)

f = open('%s-out.bat' % src, 'w')
silence_file = open('%s-silences.txt' % src, 'w')

tmprate = 22050
len2 = dur * tmprate
buflen = int(len2     * 2)
#            t * rate * 16 bits
print(buflen)

oarr = numpy.arange(1, dtype='int16')
# just a dummy array for the first chunk

command = [ FFMPEG_BIN,
        '-i', src,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(tmprate), # ouput sampling rate
        '-ac', '1', # '1' for mono
        '-']        # - output to stdout

pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

tf = True
pos = 0
opos = 0
part = 0

pos_start = 0
pos_end = 0

while tf :

    raw = pipe.stdout.read(buflen)
    if raw.decode('ISO-8859-1') == '' :
        print('End')
        tf = False
        break
    

    rng = numpy.frombuffer(raw, dtype = "int16")

    # rng = numpy.concatenate([oarr, arr])
    mx = numpy.amax(rng)
    if mx <= thr :
        # the peak in this range is less than the threshold value
        trng = (rng <= thr) * 1
        # effectively a pass filter with all samples <= thr set to 0 and > thr set to 1
        sm = numpy.sum(trng)
        # i.e. simply (naively) check how many 1's there were
        if sm >= len2 :
            pos_end = pos + dur
            part += 1
            apos = pos + dur
            # print(mx, sm, len2, apos, opos)
            # silence_file.write('{:f},{:f}\n'.format(pos, apos))

            # silence_file.write('{:f},{:f}\n'.format(pos, apos))

            f.write('ffmpeg -i "{:s}" -ss {:f} -to {:f} -c copy -y "{:s}-p{:04d}.mp3"\r\n'.format(src, pos, apos, src, part))
        elif pos_start != pos_end:
            silence_file.write('{:f},{:f}\n'.format(pos_start, pos_end))
            pos_start = pos_end
            # opos = apos

    elif pos_start != pos_end:
        silence_file.write('{:f},{:f}\n'.format(pos_start, pos_end))
        pos_start = pos_end

    pos += dur

    # oarr = arr

# part += 1    
# f.write('ffmpeg -i "%s"  -ss  %f   -to %f   -c copy -y "%s-p%04d.mp3"\r\n' % (src, opos, pos, src, part))
# silence_file.write('{:f},{:f}\n'.format(pos, apos))
f.close()