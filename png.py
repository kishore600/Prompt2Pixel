import struct
from tracemalloc import Filter
import zlib
# PNG chunk 
# +--------+------+---------+------+
# | Length | Type | Data | CRC |
# +--------+------+---------+------+
def write_chunk(f, chunk_type, data):
    f.write(struct.pack('>I', len(data)))
    f.write(chunk_type)
    f.write(data)
    crc = zlib.crc32(chunk_type + data) & 0xffffffff
    f.write(struct.pack('>I', crc))

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

def write_png(filename,width,height,pixels):
    with open(filename,'wb') as f:
        f.write(PNG_SIGNATURE)                             
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        write_chunk(f,b'IHDR',ihdr_data)
        raw = bytearray()
        for row in pixels:
            raw.append(0)  # filter type 0 (None)
            for r,g,b in row:
                raw += bytes([r,g,b])
        compressed = zlib.compress(bytes(raw))
        write_chunk(f,b'IDAT',compressed)
        write_chunk(f,b'IEND',b'')

# 1. Set up dx, dy, a bucket = 0, and x, y = x0, y0.
# 2. Start a results list, and plot the very first point (x0, y0) into it.
# 3. Loop x from x0 + 1 up to and including x1 (since we already plotted the start):
#   - add dy to the bucket
#   - if bucket >= dx: increment y, subtract dx from the bucket
#   - append (x, y) to the results list
# 4. Return the results list.
def bresenham_line_simple(x0,y0,x1,y1):
    dx = x1 - x0
    dy = y1 - y0
    bucket = 0
    x,y = x0,y0
    result = [(x0,y0)]

    for x in range(x0+1,x1+1):
        bucket = bucket + dy
        if bucket >= dx:
            y = y + 1
            bucket = bucket - dx
            result.append((x,y))
    return result

if __name__ == '__main__':
    width , height = 250,250
    red = (220,40,40)
    pixels = [[red for _ in range(width)] for _ in range(height)]
    write_png('red.png',width,height,pixels)
    result = bresenham_line_simple(0, 0, 5, 2)
    print(result)  