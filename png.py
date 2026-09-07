import struct
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

    
with open("test.bin", "wb") as f:
    write_chunk(f, b'TEST', b'hello')